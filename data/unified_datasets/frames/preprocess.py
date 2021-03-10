import zipfile
import json
import os
from pprint import pprint
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
from convlab2.util.file_util import read_zipped_json, write_zipped_json
import re
self_dir = os.path.dirname(os.path.abspath(__file__))


intent2des = {
    "inform": "Inform a slot value",
    "offer": "Offer a package to the user",
    "request": "Ask for the value of a particular slot",
    "switch_frame": "Switch to a frame",
    "suggest": "Suggest a slot value or package that does not match the user's constraints",
    "no_result": "Tell the user that the database returned no results",
    "thankyou": "Thank the other speaker",
    "sorry": "Apologize to the user",
    "greeting": "Greet the other speaker",
    "affirm": "Affirm something said by the other speaker",
    "negate": "Negate something said by the other speaker",
    "confirm": "Ask the other speaker to confirm a given slot value",
    "moreinfo": "Ask for more information on a given set of results",
    "goodbye": "Say goodbye to the other speaker",
    "request_alts": "Ask for other possibilities",
    "request_compare": "Ask the wizard to compare packages",
    "hearmore": "Ask the user if she'd like to hear more about a given package",
    "you_are_welcome": "Tell the user she is welcome",
    "canthelp": "Tell the user you cannot answer her request",
    "reject": "Tell the user you did not understand what she meant"
}

slot2des = {
    "book": "Find a trip to book",
    "dst_city": "Destination city",
    "or_city": "Origin city",
    "str_date": "Start date for the trip",
    "n_adults": "Number of adults",
    "budget": "The amount of money that the user has available to spend for the trip.",
    "end_date": "End date for the trip",
    "flex": "Boolean value indicating whether the constraints are flexible",
    "duration": "Duration of the trip",
    "ref_anaphora": "Words used to refer to a frame",
    "price": "Price of the trip including flights and hotel",
    "max_duration": "Maximum number of days for the trip",
    "amenities": "Number of amenities",
    "vicinity": "Vicinity of the hotel",
    "name": "Name of the hotel",
    "category": "Rating of the hotel (in number of stars)",
    "wifi": "Boolean value indicating whether or not the hotel offers free wifi",
    "booked": "Booked a trip",
    "dep_time_or": "Time of departure from origin city",
    "n_children": "Number of children",
    "gst_rating": "Rating of the hotel by guests (in number of stars)",
    "parking": "Boolean value indicating whether or not the hotel offers free parking",
    "arr_time_or": "Time of arrival to origin city",
    "breakfast": "Boolean value indicating whether or not the hotel offers free breakfast",
    "count": "Number of different packages",
    "seat": "Seat type (economy or business)",
    "count_name": "Number of different hotels",
    "count_dst_city": "Number of destination cities",
    "budget_ok": "Boolean value indicating whether the package fits the budget",
    "arr_time_dst": "Time of arrival to destination",
    "dep_time_dst": "Time of departure from destination",
    "gym": "Boolean value indicating whether or not the hotel offers gym",
    "spa": "Boolean value indicating whether or not the hotel offers spa",
    "downtown": "Boolean value indicating whether or not the hotel is in the heart of the city",
    "min_duration": "Minimum number of days for the trip",
    "airport": "Boolean value indicating whether or not the hotel is in the vicinity of an airport",
    "beach": "Boolean value indicating whether or not the hotel is in the vicinity of a beach",
    "museum": "Boolean value indicating whether or not the hotel is in the vicinity of a museum",
    "theatre": "Boolean value indicating whether or not the hotel is in the vicinity of a theatre",
    "park": "Boolean value indicating whether or not the hotel is in the vicinity of a park",
    "market": "Boolean value indicating whether or not the hotel is in the vicinity of a market",
    "shopping": "Boolean value indicating whether or not the hotel is in the vicinity of a shopping center",
    "university": "Boolean value indicating whether or not the hotel is in the vicinity of an university",
    "mall": "Boolean value indicating whether or not the hotel is in the vicinity of a mall",
    "palace": "Boolean value indicating whether or not the hotel is in the vicinity of a palace",
    "cathedral": "Boolean value indicating whether or not the hotel is in the vicinity of a cathedral",
    "no_result": "Boolean value indicating whether there is no result match user's constraints"
}


def get_slot_type(slot):
    if slot in {'book', 'booked', 'vicinity', 'amenities'}:
        return 'binary'
    elif slot in {'dst_city', 'or_city', 'str_date', 'end_date', 'duration', 'min_duration', 'max_duration',
                  'dep_time_or', 'arr_time_or', 'arr_time_dst', 'dep_time_dst', 'n_adults', 'n_children', 'budget',
                  'price', 'ref_anaphora', 'name', 'category', 'gst_rating',
                  'count', 'count_name', 'count_dst_city', 'seat'}:
        return 'non-categorical'
    elif slot in {'budget_ok', 'flex', 'wifi', 'parking', 'breakfast', 'gym', 'spa', 'downtown', 'airport', 'beach',
                  'museum', 'theatre', 'park', 'market', 'shopping', 'university', 'mall', 'palace', 'cathedral'}:
        return 'categorical'
    else:
        return None


digit2word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
}

match = {
    '0': 0,
    '1': 0,
    '>1': 0,
}


def pharse_in_sen(phrase, sen):
    '''
    match value in the sentence
    :param phrase: str
    :param sen: str
    :return: start, end if matched, else None, None
    '''
    assert isinstance(phrase, str)
    pw = '(^|[\s,\.:\?!-])(?P<v>{})([\s,\.:\?!-]|$)'
    pn = '(^|[\s\?!-]|\D[,\.:])(?P<v>{})($|[\s\?!-]|[,\.:]\D|[,\.:]$)'
    if phrase.isdigit():
        pattern = pn
    else:
        pattern = pw
    p = re.compile(pattern.format(re.escape(phrase)), re.I)
    m = re.search(p, sen)
    if m:
        num = len(re.findall(p, sen))
        if num > 1:
            match['>1'] += 1
        else:
            match['1'] += 1
        return m.span('v'), num
    if phrase.isdigit() and phrase in digit2word:
        phrase = digit2word[phrase]
        p = re.compile(pw.format(re.escape(phrase)), re.I)
        m = re.search(p, sen)
        if m:
            num = len(re.findall(p, sen))
            if num > 1:
                match['>1'] += 1
            else:
                match['1'] += 1
            return m.span('v'), num
    match['0'] += 1
    return (None, None), 0


def iter_over_acts(acts):
    for act in acts:
        intent = act['name']
        for arg in act['args']:
            k = arg['key']
            if k == 'id':
                continue
            elif k in ['ref', 'read', 'write']:
                assert isinstance(arg['val'], list)
                for frame in arg['val']:
                    for kv in frame['annotations']:
                        if kv['key'] in ('ref', 'read', 'write'):
                            print(kv, frame)
                            assert False
                        yield intent, kv['key'], kv.get('val')
            else:
                yield intent, k, arg.get('val', None)


def normalize_da(intent, slot, value, utterance):
    if slot == 'intent':
        slot = 'book'
    elif slot == 'action':
        slot = 'booked'
    elif slot not in slot2des:
        # ignore some rare slot
        return None, None

    if slot in ['book', 'booked']:
        slot_type = 'binary'
        return slot_type, {
            "intent": intent,
            "domain": 'travel',
            "slot": slot,
            "value": 'True',
        }
    elif value is None or value == '':
        slot_type = 'binary'
        return slot_type, {
            "intent": intent,
            "domain": 'travel',
            "slot": slot,
            "value": '',
        }
    elif value == '-1':
        slot_type = 'binary'
        return slot_type, {
            "intent": intent,
            "domain": 'travel',
            "slot": slot,
            "value": 'dontcare',
        }
    elif isinstance(value, str):
        slot_type = get_slot_type(slot)
        assert slot_type == 'non-categorical'
        (start, end), num = pharse_in_sen(value, utterance)
        if not num:
            if slot == 'gst_rating' and pharse_in_sen(' / '.join(value.split('/')), utterance)[1]:
                value = ' / '.join(value.split('/'))
            elif 'a. m' in value and pharse_in_sen(value.replace('a. m', 'a.m'), utterance)[1]:
                value = value.replace('a. m', 'a.m')
            elif 'p. m' in value and pharse_in_sen(value.replace('p. m', 'p.m'), utterance)[1]:
                value = value.replace('p. m', 'p.m')
            elif slot == 'price' and pharse_in_sen(value.replace('USD', ' USD'), utterance)[1]:
                value = value.replace('USD', ' USD')
            else:
                # few wrong annotation
                return None, None
            (start, end), num = pharse_in_sen(value, utterance)
            assert num, print(value, utterance)
            if not num:
                return None, None
            # return None, None
        return slot_type, {
            "intent": intent,
            "domain": 'travel',
            "slot": slot,
            "value": utterance[start:end],
            "start": start,
            "end": end
        }
    elif isinstance(value, bool):
        slot_type = get_slot_type(slot)
        value = str(value)
        assert slot_type == 'categorical' or slot_type == 'binary', print(slot, value)
        return slot_type, {
            "intent": intent,
            "domain": 'travel',
            "slot": slot,
            "value": value,
        }
    else:
        assert 0


def preprocess():
    processed_dialogue = []
    ontology = {'domains': {'travel':
                                {"description": "Book a vacation package containing round-trip flights and a hotel.",
                                 "slots": {}}},
                'intents': {},
                'binary_dialogue_act': [],
                'state': {}}
    original_zipped_path = os.path.join(self_dir, 'original_data.zip')
    new_dir = os.path.join(self_dir, 'original_data')
    if not os.path.exists(original_zipped_path):
        raise FileNotFoundError(original_zipped_path)
    if not os.path.exists(os.path.join(self_dir, 'data.zip')) or not os.path.exists(os.path.join(self_dir, 'ontology.json')):
        print('unzip to', new_dir)
        print('This may take several minutes')
        archive = zipfile.ZipFile(original_zipped_path, 'r')
        archive.extractall(new_dir)
        data = json.load(open(os.path.join(new_dir, 'frames.json')))
        # json.dump(data, open(os.path.join(new_dir, 'original_data.json'), 'w'), indent=2)
        cnt = 1
        for d in tqdm(data, desc='dialogue'):
            dialogue = {
                "dataset": 'frames',
                "data_split": 'train',
                "dialogue_id": 'frames_' + str(cnt),
                "original_id": d['id'],
                "user_id": d['user_id'],
                "system_id": d['wizard_id'],
                "userSurveyRating": d['labels']['userSurveyRating'],
                "wizardSurveyTaskSuccessful": d['labels']['wizardSurveyTaskSuccessful'],
                "domains": ['travel'],
                "turns": []
            }
            # state = deepcopy(ontology['state']['travel'])
            for utt_idx, t in enumerate(d['turns']):
                speaker = 'system' if t['author']=='wizard' else t['author']
                turn = {
                    'speaker': speaker,
                    'utterance': t['text'],
                    'utt_idx': utt_idx,
                    'dialogue_act': {
                        'binary': [],
                        'categorical': [],
                        'non-categorical': [],
                    },
                }
                for intent, slot, value in iter_over_acts(t['labels']['acts']):
                    da_type, da = normalize_da(intent, slot, value, t['text'])
                    if da is not None:
                        da['value'] = da['value'].lower()
                        turn['dialogue_act'][da_type].append(da)
                        slot = da['slot']
                        value = da['value']
                        if da_type == 'binary':
                            if da not in ontology['binary_dialogue_act']:
                                ontology['binary_dialogue_act'].append(da)
                        else:
                            ontology['domains']['travel']['slots'].setdefault(slot, {
                                "description": slot2des[slot],
                                "is_categorical": da_type=='categorical',
                                "possible_values": []
                            })
                            if da_type == 'categorical' \
                                    and value not in ontology['domains']['travel']['slots'][slot]['possible_values']:
                                ontology['domains']['travel']['slots'][slot]['possible_values'].append(value)
                        ontology['intents'].setdefault(intent, {
                            "description": intent2des[intent]
                        })
                # state
                if speaker == 'user':
                    turn['state'] = {}
                    turn['state_update'] = {
                        'categorical': [],
                        'non-categorical': [],
                    }
                dialogue['turns'].append(deepcopy(turn))
            cnt += 1
            if len(dialogue['turns']) % 2 == 0:
                dialogue['turns'] = dialogue['turns'][:-1]
            processed_dialogue.append(deepcopy(dialogue))
        ontology['binary_dialogue_act'] = sorted(ontology['binary_dialogue_act'], key=lambda x: x['intent'])
        json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w'), indent=2)
        json.dump(processed_dialogue, open('data.json', 'w'), indent=2)
        write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        os.remove('data.json')
    else:
        # read from file
        processed_dialogue = read_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        ontology = json.load(open(os.path.join(self_dir, 'ontology.json')))
    return processed_dialogue, ontology


if __name__ == '__main__':
    preprocess()
    print(match) # {'0': 271, '1': 29333, '>1': 806}
