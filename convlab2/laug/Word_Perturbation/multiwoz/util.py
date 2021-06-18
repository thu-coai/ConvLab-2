import re
import json
import os
import random
import string
from copy import deepcopy
from typing import List, Iterable
from collections import defaultdict, Counter
from functools import reduce, lru_cache
from io import StringIO
from pprint import pprint

from .types import MultiwozSampleType, MultiwozDatasetType, SentenceType
from .tokenize_util import tokenize, convert_sentence_to_tokens


## load json and dump json ############
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_json(obj, filepath, **json_kwargs):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json_kwargs.setdefault('indent', 4)
        json.dump(obj, f, **json_kwargs)


## better str ##################################
def p_str(obj):
    sio = StringIO()
    pprint(obj, sio)
    return sio.getvalue()


punctuation_pattern = re.compile(r'^[{}]+$'.format('\\'.join(string.punctuation)))
RePatternType = type(re.compile(''))


def is_punctuation(word):
    return punctuation_pattern.match(word) is not None


## Helper class ###################
def _get_all_slots(multiwoz: MultiwozDatasetType):
    slots = defaultdict(dict)  # Dict[str, Dict[str, Set[str]]]; Dict[Sample_ID, Dict[intent, Set[Slot]]]
    for sample in multiwoz.values():
        logs = sample['log']
        if isinstance(logs, dict):
            logs = logs.values()
        for turn in logs:
            dialog_act = turn['dialog_act']
            for domain_intent, slot_value_list in dialog_act.items():
                domain, intent = domain_intent.lower().split('-')
                for slot, value in slot_value_list:
                    slots[domain].setdefault(intent, set()).add(slot.lower() if isinstance(slot, str) else slot)
    for domain in slots:
        s = reduce((lambda s1, s2: s1 | s2), slots[domain].values(), set())
        slots[domain]['all'] = s
    for domain in slots:
        unique_slots = slots[domain]['all'].copy()
        for other_domain in slots:
            if other_domain != domain:
                unique_slots -= slots[other_domain]['all']
        slots[domain]['unique'] = unique_slots
    return slots


class Patterns:
    time_pattern = re.compile(r"^\d{1,2}:\d{1,2}$")
    integer_pattern = re.compile(r"^[+-]?(\d+|\d{1,3}(,\d{3})*)$")
    ref_pattern = re.compile(r"(?=.*[A-Z].*[0-9].*|.*[0-9].*[A-Z].*)[A-Z0-9]{8}")


class Helper:
    def __init__(self, multiwoz):
        self.multiwoz = multiwoz
        self.slots = _get_all_slots(multiwoz)

    def get_unique_slots(self, domain):
        return self.slots[domain]['unique']

    @staticmethod
    @lru_cache(1000)
    def split_str(s):
        return s.split()

    _words_about_slot = {
        'attraction': {
            'area': 'area', 'type': 'type', 'name': 'name',
            'fee': ['entrance fee'], 'addr': 'address',
            'post': ['postcode', 'post code'], 'phone': 'phone'
        },
        'hospital': {
            'department': 'department', 'addr': 'address', 'post': ['postcode', 'post code'],
            'phone': 'phone'
        },
        'hotel': {
            'type': 'type', 'parking': 'parking', 'price': ['pricerange', 'price range'],
            'internet': ['internet', 'wifi'], 'area': 'area', 'stars': 'stars',
            'name': 'name', 'stay': ['stay', Patterns.integer_pattern], 'day': 'day',
            'people': ['people', Patterns.integer_pattern],
            'addr': 'address', 'post': ['postcode', 'post code'], 'phone': 'phone'
        },
        'police': {
            'addr': 'address', 'post': ['postcode', 'post code'], 'phone': 'phone', 'name': 'name'
        },
        'restaurant': {
            'food': 'food', 'price': ['pricerange', 'price range'], 'area': 'area',
            'name': 'name', 'time': 'time', 'day': 'day', 'people': 'people',
            'phone': 'phone', 'post': ['postcode', 'post code'], 'addr': 'address'
        },
        'taxi': {
            'leave': ['leaveat', 'leave at', Patterns.time_pattern], "dest": ['destination', "cambridge", 'belfry'],
            'depart': 'departure',
            'arrive': ['arriveby', 'arrive by', Patterns.time_pattern],
            'car': 'car type', 'phone': 'phone'
        },
        'train': {
            'dest': ['destination', "cambridge"], 'day': 'day',
            'arrive': ['arriveby', 'arrive by', Patterns.time_pattern],
            'depart': 'departure', 'leave': ['leaveat', 'leave at', Patterns.time_pattern], 'people': 'people',
            'time': 'duration', 'id': 'trainid'
        }
    }

    def relevant_words_of_slot(self, domain, slot):
        if domain not in self._words_about_slot or slot not in self._words_about_slot[domain]:
            return [slot]
        if isinstance(self._words_about_slot[domain][slot], str):
            res = [self._words_about_slot[domain][slot]]
            if slot != res[-1]:
                res.append(slot)
            return res
        else:
            return self._words_about_slot[domain][slot] + [slot]

    _words_about_domain = {
        'police': ['assistance'],
        'attraction': ['attractions', 'trip', 'gallery', 'museum', 'theatre', 'visit', 'entertainment', 'cinema',
                       'park', 'cambridge', 'college', 'architecture'],
        'hotel': ['place to stay', 'hotels', 'guesthouse'],
        'restaurant': ['place to eat', 'place to dine', 'food', 'gastropub', 'restaurants'],
        'booking': ['book'],
    }

    def relevant_words_of_domain(self, domain):
        if domain not in self._words_about_domain:
            return [domain]
        return self._words_about_domain[domain] + [domain]

    @staticmethod
    def contain_word(sentence: str, word):
        if isinstance(word, str):
            if not word.startswith(r'\b'):
                word = r'\b' + word
            if not word.endswith(r'\b'):
                word += r'\b'
            word = re.compile(word)
        else:
            assert isinstance(word, RePatternType)
        return word.search(sentence) is not None

    def _get_excluding_indexes(self, words, span_info, dialog_act):
        """exclude some words, so that the label keeps the same after augmented."""
        excluding_indexes = set()
        domains = set()
        slots = set()
        for domain_intent, slot, value, start, end in span_info:
            excluding_indexes.update(range(start, end + 1))
            domain = domain_intent.split('-')[0].lower()
            domains.add(domain)
            slots.add((domain, slot.lower()))  
        for domain_intent, slot_value_list in dialog_act.items():
            domain = domain_intent.split('-')[0].lower()
            domains.add(domain)
            for slot, value in slot_value_list:
                slots.add((domain, slot.lower() if isinstance(slot, str) else slot))

        word2index = {v.lower(): i for i, v in enumerate(words)}
        for domain in domains:
            for word in self.relevant_words_of_domain(domain):
                if isinstance(word, str):
                    ws = tokenize(word)
                    if len(ws) == 1:
                        if ws[0] in word2index:
                            excluding_indexes.add(word2index[word])
                    else:
                        n = len(ws)
                        N = len(words)
                        for i in range(N):
                            if i + n <= N and all(ws[j] == words[i + j] for j in range(n)):
                                excluding_indexes.update(range(i, i + n))

                if isinstance(word, RePatternType):
                    for i in range(len(words)):
                        if word.match(words[i]):
                            excluding_indexes.add(i)
        for domain, slot in slots:
            for word in self.relevant_words_of_slot(domain, slot):
                if isinstance(word, str):
                    ws = tokenize(word)
                    if len(ws) == 1:
                        if ws[0] in word2index:
                            excluding_indexes.add(word2index[word])
                    else:
                        n = len(ws)
                        N = len(words)
                        for i in range(N):
                            if i + n <= N and all(ws[j] == words[i + j] for j in range(N)):
                                excluding_indexes.update(range(i, i + n))

                if isinstance(word, RePatternType):
                    for i in range(len(words)):
                        if word.match(words[i]):
                            excluding_indexes.add(i)

        for i, word in enumerate(words):
            if is_punctuation(word):
                excluding_indexes.add(i)
            elif word == 'reference' and i + 1 < len(words) and words[i + 1] == 'number':
                # exclude "reference number"
                excluding_indexes.update((i, i + 1))
        return excluding_indexes


## iter dialogues ############
## the data format of the augmented multiwoz may be different from the original multiwoz
def _iter_dialogues(sample: MultiwozSampleType):
    dialogues = sample['log']
    if isinstance(dialogues, list):
        for i, dialog in enumerate(dialogues):
            turn = dialog.get('turn', i)
            yield turn, dialog
    elif isinstance(dialogues, dict):
        # assume key is `turn`
        yield from dialogues.items()
    else:
        raise RuntimeError("unknown format.")


def iter_dialogues(sample: MultiwozSampleType, mode='usr'):
    assert mode in ('usr', 'user', 'all', 'sys')
    for turn, dialog in _iter_dialogues(sample):
        if mode in ("usr", 'user') and turn % 2 == 1:
            continue
        if mode == 'sys' and turn % 2 == 0:
            continue
        yield turn, dialog


## random choice ####################
_EmptySequence = object()


def choice(seq: Iterable):
    if hasattr(seq, '__len__') and hasattr(seq, '__getitem__'):
        return random.choice(seq)

    r = _EmptySequence
    for i, x in enumerate(seq, 1):
        if random.random() * i <= 1:
            r = x
    if r is _EmptySequence:
        raise ValueError("empty sequence")
    return r


## record augmented text and span info, then returns an augmented sample
class AugmentationRecorder:
    def __init__(self, original_sample: MultiwozSampleType):
        self.original_sample = original_sample
        self.augmented_turns = []

    def add_augmented_dialog(self, turn_index, turn):
        self.augmented_turns.append((turn_index, turn))

    def get_augmented_sample(self) -> MultiwozSampleType:
        sample = deepcopy(self.original_sample)
        turns = sample['log']
        counter = Counter()
        for turn_index, turn in self.augmented_turns:
            # if there is more than one augmented text
            # random choose one
            counter[turn_index] += 1
            if random.random() * counter[turn_index] <= 1:
                turns[turn_index] = {'turn_index': turn_index, 'augmented': True, **turn}
        return sample


## check whether span info is consistent with text
def _equal_words(words1, words2, ignore_case):
    if not ignore_case:
        return words1 == words2
    else:
        return len(words1) == len(words2) and all(w1.lower() == w2.lower() for w1, w2 in zip(words1, words2))


def is_span_info_consistent_with_text(sentence: SentenceType, span_info: List[list], ignore_case=True) -> bool:
    """check whether the span info is consistent with text."""
    words = convert_sentence_to_tokens(sentence)
    return all(
        _equal_words(words[start:end + 1], tokenize(span), ignore_case) for domain_intent, slot, span, start, end in
        span_info) and len({tuple(x[-2:]) for x in span_info}) == len(span_info)
