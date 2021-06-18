import random
import re
from typing import List, Dict, Optional, Union, Iterable
from copy import deepcopy
from collections import defaultdict, namedtuple

from .db_loader import BaseDBLoader, DBLoader
from .db import BaseDB, DB, choice
from ..tokenize_util import tokenize
from ..util import p_str

MultiSourceDBLoaderArgs = namedtuple('MultiSourceDBLoaderArgs', 'db_dir domain_slot_map')


class MultiSourceDBLoader(BaseDBLoader):
    @staticmethod
    def _parse_init_args(args) -> List[MultiSourceDBLoaderArgs]:
        assert isinstance(args, (list, tuple))
        if isinstance(args, MultiSourceDBLoaderArgs):
            return [args]

        def toMultiSourceDBLoaderArgs(arg):
            if isinstance(arg, MultiSourceDBLoaderArgs):
                return arg
            assert isinstance(arg, (list, tuple))
            assert len(arg) == len(MultiSourceDBLoaderArgs._fields)
            return MultiSourceDBLoaderArgs(*arg)

        args = [toMultiSourceDBLoaderArgs(arg) for arg in args]
        return args

    def __init__(self, args: Union[List[MultiSourceDBLoaderArgs], List[tuple], MultiSourceDBLoaderArgs]):
        self.loaders_and_maps = []
        args = self._parse_init_args(args)
        for db_dir, domain_slot_map in args:
            loader = DBLoader(db_dir)
            self.loaders_and_maps.append((loader, domain_slot_map))

    def load_db(self, domain, slot: Optional[str] = None) -> Optional["MultiSourceDB"]:
        dbs = []
        for loader, domain_slot_map in self.loaders_and_maps:
            if slot is not None:
                if (domain.lower(), slot.lower()) in domain_slot_map:
                    db_domain, db_slot = domain_slot_map[(domain.lower(), slot.lower())]
                    db = loader.load_db(db_domain, db_slot)
                    if db is not None:
                        dbs.append((db, db_domain, domain_slot_map))
            else:
                domain_to_db = {}
                for domain_slot_tuple, db_domain_slot_tuple in domain_slot_map.items():
                    if domain.lower() == domain_slot_tuple[0].lower():
                        db_domain = db_domain_slot_tuple[0]
                        if db_domain not in domain_to_db:
                            db = loader.load_db(db_domain)
                            if db is not None:
                                domain_to_db[db_domain] = db
                dbs.extend((db, db_domain, domain_slot_map) for db_domain, db in domain_to_db.items())

        if not dbs:
            return None
        return MultiSourceDB(dbs)


MultiSourceDBArgs = namedtuple('MultiSourceDBArgs', 'db db_domain domain_slot_map')


class MultiSourceDB(BaseDB):
    @staticmethod
    def _parse_init_args(args) -> List[MultiSourceDBArgs]:
        if isinstance(args, MultiSourceDBArgs):
            return [args]
        assert isinstance(args, (list, tuple))

        def toMultiSourceDBArgs(arg):
            if isinstance(arg, MultiSourceDBArgs):
                return arg
            assert isinstance(arg, (list, tuple))
            assert len(arg) == len(MultiSourceDBArgs._fields)
            return MultiSourceDBArgs(*arg)

        args = [toMultiSourceDBArgs(arg) for arg in args]
        return args

    def __init__(self, args: Union[MultiSourceDBArgs, List[MultiSourceDBArgs], List[tuple]]):
        self.args = self._parse_init_args(args)

    def find_different_values(self, domain, slot, excluding_values=()) -> Iterable:
        """find different values, which belong to the same domain and slot."""
        for db, db_domain, domain_slot_map in self.args:
            k = (domain.lower(), slot.lower())
            if k not in domain_slot_map:
                continue
            if domain_slot_map[k][0] != db_domain:
                continue
            db_domain, db_slot = domain_slot_map[k]
            r = db.query(
                lambda item: db_slot in item and item[db_slot] not in excluding_values)
            yield from (dict_[db_slot] for dict_ in r)

    def sample_value(self, domain, slot, excluding_values=()):
        values = self.find_different_values(domain, slot, excluding_values=excluding_values)
        try:
            return choice(values)
        except ValueError:
            return None


def _get_word2indexes(words, to_lower=False):
    word2indexes = defaultdict(list)
    for i, word in enumerate(words):
        if to_lower:
            word2indexes[word.lower()].append(i)
        else:
            word2indexes[word].append(i)
    return word2indexes


def _get_positions(words: List[str], word_to_indexes: Dict[str, List[int]], value: List[str]):
    first_word = value[0]
    N = len(value)
    for first_index in word_to_indexes.get(first_word, ()):
        if words[first_index: first_index + N] == value:
            return [first_index, first_index + N - 1]


def fix_text(text):
    # strip; split punctuation and word
    text = re.sub(r"(?:^|(?<=\s))([\w$']+)([,.?*/!;<=>\]\"]+)(?:$|(?=[A-Z\s]))", r'\1 \2',
                  text)  # split word and punctuation
    return text


def fix_turn(turn: dict):
    # fix error in a turn
    # turn = {
    #     'text': ...,
    #     'span_info': ...,
    #     'dialog_act': ...
    # }
    text = turn['text']
    words = tokenize(text)
    word2indexes = _get_word2indexes(words, to_lower=False)
    span_info = turn['span_info']
    dialog_act = turn['dialog_act']
    for i, item in enumerate(span_info):
        domain_intent, slot, value, *positions = item
        assert len(positions) == 2
        domain, intent = domain_intent.split('-')
        if ' '.join(words[positions[0]: positions[1] + 1]) != value:
            positions = None
        if positions is None:
            positions = _get_positions(words, word2indexes, tokenize(value))
        if positions is None:
            slot_value_list = dialog_act[domain_intent]
            for i in range(len(slot_value_list)):
                if slot_value_list[i][0] == slot:
                    value = slot_value_list[i][1]
                    break
            positions = _get_positions(words, word2indexes, tokenize(value))
        if positions is None:
            raise ValueError(f"turn: {p_str(turn)}\nitem: {p_str(item)}\nwords: {p_str(words)}\n"
                             f"word2indexes {p_str(word2indexes)}\nvalue: {tokenize(value)}")
        value = ' '.join(words[positions[0]:1 + positions[1]])
        item[2] = value
        if item[-2:] != positions:
            item[-2:] = positions
        if domain_intent not in dialog_act:
            continue
        slot_value_list = dialog_act[domain_intent]
        for i in range(len(slot_value_list)):
            if slot_value_list[i][0] == slot:
                slot_value_list[i][1] = value
    span_info.sort(key=lambda item: item[-2:])


def assert_correct_turn(turn: dict):
    text = turn['text']
    words = tokenize(text)
    span_info = turn['span_info']
    dialog_act = turn['dialog_act']
    new_dialog_act = {}
    for item in span_info:
        domain_intent, slot, value, begin, end = item
        assert words[begin: 1 + end] == tokenize(value), f"turn: {p_str(turn)}\nitem: {item}"
        new_dialog_act.setdefault(domain_intent, [])
        new_dialog_act[domain_intent].append([slot, value])
    for domain_intent, new_slot_value_list in new_dialog_act.items():
        assert domain_intent in dialog_act
        new_slot_value_set = {tuple(slot_value) for slot_value in new_slot_value_list}
        slot_value_list = dialog_act[domain_intent]
        slot_value_set = {tuple(slot_value) for slot_value in slot_value_list}
        assert new_slot_value_set <= slot_value_set, p_str([turn, new_dialog_act])
        diff = slot_value_set - new_slot_value_set
        assert all(slot == 'none' or value == '?' for slot, value in diff), f"Error, {p_str(turn)}\n{p_str(diff)}"


def replace_slot_values_in_turn(turn: dict, db_loader: MultiSourceDBLoader,
                                p=0.25,
                                inform_intents=('inform',)):
    orig_turn = turn
    turn = deepcopy(orig_turn)
    try:
        fix_turn(turn)
        assert_correct_turn(turn)
    except:
        return orig_turn
    text = turn['text']
    words = tokenize(text)
    span_info = turn['span_info']
    span_info.sort(key=lambda item: item[-2:])
    dialog_act = turn['dialog_act']
    if any(span_info[i][-2] <= span_info[i - 1][-1] for i in range(1, len(span_info))):
        return turn

    new_turn = deepcopy(turn)
    new_words = words.copy()
    new_span_info = new_turn['span_info']
    new_dialog_act = new_turn['dialog_act']
    updated_span = []

    for i, item in enumerate(span_info):
        domain_intent = item[0]
        domain, intent = domain_intent.split('-')
        slot = item[1]
        value = item[2]
        if intent.lower() not in inform_intents:
            continue
        if updated_span:
            j = updated_span[-1]
            last_item = span_info[j]
            if item[-2] <= last_item[-1]:
                continue
        db = db_loader.load_db(domain, slot)
        if db is None:
            continue
        new_value = db.sample_value(domain, slot, excluding_values=(value, 'none', '?'))
        if new_value is None:
            continue
        if random.random() > p:
            continue
        new_value = fix_text(new_value)
        new_span_info[i][2] = new_value
        new_slot_value_list = new_dialog_act[domain_intent]
        for j in range(len(new_slot_value_list)):
            if new_slot_value_list[j][0] == slot:
                new_slot_value_list[j][1] = new_value
        updated_span.append(i)
        # print(f'replace {item[2]} with {new_value}')

    # update new_words and span in new_span_info
    if updated_span:
        offset = 0
        for i in range(len(span_info)):
            begin, end = span_info[i][-2:]
            new_value = new_span_info[i][2]
            new_value = tokenize(new_value)
            num_words = len(new_value)
            new_words[offset + begin: offset + end + 1] = new_value
            new_span_info[i][-2:] = [begin + offset, begin + offset + num_words - 1]
            offset += num_words - (end - begin + 1)
        new_turn['text'] = ' '.join(new_words)
        assert_correct_turn(new_turn)
    return new_turn


def replace_slot_values(sample, db_loader: MultiSourceDBLoader,
                        p=0.25,
                        inform_intents=('inform',),
                        mode='usr'):
    """
    replace slot values in a sample

    Args:
        sample: a dialogue
        db_loader: it can loads a db
        p: probability to replace if conditions are satisfied
        inform_intents: only inform intents may be replaced.
        mode: 'usr' or 'user': only replace on user turns;
              'sys': on;y replace on sys turns;
              'all': replace on all turns
    """
    new_sample = deepcopy(sample)
    for turn_index, turn in enumerate(sample['log']):
        is_user = turn_index % 2 == 0
        if is_user and mode not in ('usr', 'user', 'all'):
            continue
        if not is_user and mode not in ('sys', 'system', 'all'):
            continue
        new_turn = replace_slot_values_in_turn(turn, db_loader, p=p, inform_intents=inform_intents)
        new_sample['log'][turn_index] = new_turn
    return new_sample
