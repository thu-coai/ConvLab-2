import zipfile
import json
import os
from pprint import pprint
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
import numpy as np
from convlab2.util.file_util import read_zipped_json, write_zipped_json
import re
self_dir = os.path.dirname(os.path.abspath(__file__))


norm_service2domain = {
    'alarm': 'alarm',
    'banks': 'bank',
    'buses': 'bus',
    'calendar': 'calendar',
    'events': 'event',
    'flights': 'flight',
    'homes': 'home',
    'hotels': 'hotel',
    'media': 'media',
    'messaging': 'messaging',
    'movies': 'movie',
    'music': 'music',
    'payment': 'payment',
    'rentalcars': 'rentalcar',
    'restaurants': 'restaurant',
    'ridesharing': 'ridesharing',
    'services': 'services',
    'trains': 'train',
    'travel': 'travel',
    'weather': 'weather'
}

digit2word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
}

match = {
    '0': 0,
    '1': 0,
    '>1': 0,
}


def service2domain(service):
    s, i = service.split('_')
    return norm_service2domain[s.lower()]+'_'+i


def slot_normalize(service, slot):
    pass


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


def number_in_sen(word, sen):
    if ' '+word+' ' in sen:
        return sen.index(' ' + word + ' ') + 1, sen.index(' ' + word + ' ') + 1 + len(word)
    elif ' '+word+'.' in sen:
        return sen.index(' ' + word + '.') + 1, sen.index(' ' + word + '.') + 1 + len(word)
    elif ' '+word+',' in sen:
        return sen.index(' ' + word + ',') + 1, sen.index(' ' + word + ',') + 1 + len(word)
    elif sen.startswith(word+ ' ') or sen.startswith(word+'.') or sen.startswith(word+','):
        return 0, len(word)
    elif word.isdigit() and word in digit2word:
        ori_word = word
        ori_sen = sen
        word = digit2word[word]
        sen = sen.lower()
        if ' ' + word + ' ' in sen:
            return sen.index(' ' + word + ' ') + 1, sen.index(' ' + word + ' ') + 1 + len(word)
        elif ' ' + word + '.' in sen:
            return sen.index(' ' + word + '.') + 1, sen.index(' ' + word + '.') + 1 + len(word)
        elif ' ' + word + ',' in sen:
            return sen.index(' ' + word + ',') + 1, sen.index(' ' + word + ',') + 1 + len(word)
        elif sen.startswith(word + ' ') or sen.startswith(word + '.') or sen.startswith(word + ','):
            return 0, len(word)
        word = ori_word
        sen = ori_sen
    return sen.index(word)


def sys_intent():
    return {
        "inform": {"description": "Inform the value for a slot to the user."},
        "request": {"description": "Request the value of a slot from the user."},
        "confirm": {"description": "Confirm the value of a slot before making a transactional service call."},
        "offer": {"description": "Offer a certain value for a slot to the user."},
        "notify_success": {"description": "Inform the user that their request was successful."},
        "notify_failure": {"description": "Inform the user that their request failed."},
        "inform_count": {"description": "Inform the number of items found that satisfy the user's request."},
        "offer_intent": {"description": "Offer a new intent to the user."},
        "req_more": {"description": "Asking the user if they need anything else."},
        "goodbye": {"description": "End the dialogue."},
    }


def usr_intent():
    return {
        "inform_intent": {"description": "Express the desire to perform a certain task to the system."},
        "negate_intent": {"description": "Negate the intent which has been offered by the system."},
        "affirm_intent": {"description": "Agree to the intent which has been offered by the system."},
        "inform": {"description": "Inform the value of a slot to the system."},
        "request": {"description": "Request the value of a slot from the system."},
        "affirm": {"description": "Agree to the system's proposition. "},
        "negate": {"description": "Deny the system's proposal."},
        "select": {"description": "Select a result being offered by the system."},
        "request_alts": {"description": "Ask for more results besides the ones offered by the system."},
        "thank_you": {"description": "Thank the system."},
        "goodbye": {"description": "End the dialogue."},
    }


def get_intent():
    """merge sys & usr intent"""
    return {
        "inform": {"description": "Inform the value for a slot."},
        "request": {"description": "Request the value of a slot."},
        "confirm": {"description": "Confirm the value of a slot before making a transactional service call."},
        "offer": {"description": "Offer a certain value for a slot to the user."},
        "notify_success": {"description": "Inform the user that their request was successful."},
        "notify_failure": {"description": "Inform the user that their request failed."},
        "inform_count": {"description": "Inform the number of items found that satisfy the user's request."},
        "offer_intent": {"description": "Offer a new intent to the user."},
        "req_more": {"description": "Asking the user if they need anything else."},
        "goodbye": {"description": "End the dialogue."},
        "inform_intent": {"description": "Express the desire to perform a certain task to the system."},
        "negate_intent": {"description": "Negate the intent which has been offered by the system."},
        "affirm_intent": {"description": "Agree to the intent which has been offered by the system."},
        "affirm": {"description": "Agree to the system's proposition. "},
        "negate": {"description": "Deny the system's proposal."},
        "select": {"description": "Select a result being offered by the system."},
        "request_alts": {"description": "Ask for more results besides the ones offered by the system."},
        "thank_you": {"description": "Thank the system."},
    }


def preprocess():
    processed_dialogue = []
    ontology = {'domains': {},
                'intents': {},
                'binary_dialogue_act': [],
                'state': {}}
    ontology['intents'].update(get_intent())
    numerical_slots = {}
    original_zipped_path = os.path.join(self_dir, 'original_data.zip')
    new_dir = os.path.join(self_dir, 'original_data')
    if not os.path.exists(original_zipped_path):
        raise FileNotFoundError(original_zipped_path)
    if not os.path.exists(os.path.join(self_dir, 'data.zip')) or not os.path.exists(os.path.join(self_dir, 'ontology.json')):
        print('unzip to', new_dir)
        print('This may take several minutes')
        archive = zipfile.ZipFile(original_zipped_path, 'r')
        archive.extractall(self_dir)
        cnt = 1
        non_cate_slot_update_cnt = 0
        non_cate_slot_update_fail_cnt = 0
        state_cnt = {}
        num_train_dialog = 0
        num_train_utt = 0
        for data_split in ['train', 'dev', 'test']:
            dataset_name = 'schema'
            data_dir = os.path.join(new_dir, data_split)
            # schema => ontology
            f = open(os.path.join(data_dir, 'schema.json'))
            data = json.load(f)
            for schema in data:
                domain = service2domain(schema['service_name'])
                ontology['domains'].setdefault(domain, {})
                ontology['domains'][domain]['description'] = schema['description']
                ontology['domains'][domain].setdefault('slots', {})
                ontology['state'].setdefault(domain, {})
                for slot in schema['slots']:
                    # numerical => non-categorical: not use
                    # is_numerical = slot['is_categorical']
                    # for value in slot['possible_values']:
                    #     if not value.isdigit():
                    #         is_numerical = False
                    #         break
                    # if is_numerical:
                    #     numerical_slots.setdefault(slot['name'].lower(), 1)
                    lower_values = [x.lower() for x in slot['possible_values']]
                    ontology['domains'][domain]['slots'][slot['name'].lower()] = {
                        "description": slot['description'],
                        "is_categorical": slot['is_categorical'],
                        "possible_values": lower_values
                    }
                    ontology['state'][domain][slot['name'].lower()] = ''
                # add 'count' slot
                ontology['domains'][domain]['slots']['count'] = {
                    "description": "the number of items found that satisfy the user's request.",
                    "is_categorical": False,
                    "possible_values": []
                }
                # ontology['state'][domain]['count'] = ''
            # pprint(numerical_slots)
            # dialog
            for root, dirs, files in os.walk(data_dir):
                fs = sorted([x for x in files if 'dialogues' in x])
                for f in tqdm(fs, desc='processing schema-guided-{}'.format(data_split)):
                    data = json.load(open(os.path.join(data_dir, f)))
                    if data_split == 'train':
                        num_train_dialog += len(data)
                    for d in data:
                        dialogue = {
                            "dataset": dataset_name,
                            "data_split": data_split if data_split!='dev' else 'val',
                            "dialogue_id": dataset_name+'_'+str(cnt),
                            "original_id": d['dialogue_id'],
                            "domains": [service2domain(s) for s in d['services']],
                            "turns": []
                        }
                        # if d['dialogue_id'] != '84_00008':
                        #     continue
                        cnt += 1
                        prev_sys_frames = []
                        prev_user_frames = []
                        all_slot_spans_from_da = []
                        state = {}
                        for domain in dialogue['domains']:
                            state.setdefault(domain, deepcopy(ontology['state'][domain]))
                        if data_split == 'train':
                            num_train_utt += len(d['turns'])
                        for utt_idx, t in enumerate(d['turns']):
                            speaker = t['speaker'].lower()
                            turn = {
                                'speaker': speaker,
                                'utterance': t['utterance'],
                                'utt_idx': utt_idx,
                                'dialogue_act': {
                                    'binary': [],
                                    'categorical': [],
                                    'non-categorical': [],
                                },
                            }
                            for i, frame in enumerate(t['frames']):
                                domain = service2domain(frame['service'])
                                for action in frame['actions']:
                                    intent = action['act'].lower()
                                    assert intent in ontology['intents'], [intent]
                                    slot = action['slot'].lower()
                                    value_list = action['values']
                                    if action['act'] in ['REQ_MORE', 'AFFIRM', 'NEGATE', 'THANK_YOU', 'GOODBYE']:
                                        turn['dialogue_act']['binary'].append({
                                            "intent": intent,
                                            "domain": '',
                                            "slot": '',
                                            "value": '',
                                        })
                                    elif action['act'] in ['NOTIFY_SUCCESS', 'NOTIFY_FAILURE', 'REQUEST_ALTS', 'AFFIRM_INTENT', 'NEGATE_INTENT']:
                                        # Slot and values are always empty
                                        turn['dialogue_act']['binary'].append({
                                            "intent": intent,
                                            "domain": domain,
                                            "slot": '',
                                            "value": '',
                                        })
                                    elif action['act'] in ['OFFER_INTENT', 'INFORM_INTENT']:
                                        # always has "intent" as the slot, and a single value containing the intent being offered.
                                        assert slot == 'intent'
                                        turn['dialogue_act']['binary'].append({
                                            "intent": intent,
                                            "domain": domain,
                                            "slot": slot,
                                            "value": value_list[0].lower(),
                                        })
                                    elif action['act'] in ['REQUEST', 'SELECT'] and not value_list:
                                        # always contains a slot, but values are optional.
                                        # assert slot in ontology['domains'][domain]['slots']
                                        turn['dialogue_act']['binary'].append({
                                            "intent": intent,
                                            "domain": domain,
                                            "slot": slot,
                                            "value": '',
                                        })
                                    elif action['act'] in ['INFORM_COUNT']:
                                        # always has "count" as the slot, and a single element in values for the number of results obtained by the system.
                                        value = value_list[0]
                                        assert slot in ontology['domains'][domain]['slots']
                                        (start, end), num = pharse_in_sen(value, t['utterance'])
                                        if num:
                                            assert value.lower() == t['utterance'][start:end].lower() \
                                                   or digit2word[value].lower() == t['utterance'][start:end].lower()
                                            turn['dialogue_act']['non-categorical'].append({
                                                "intent": intent,
                                                "domain": domain,
                                                "slot": slot.lower(),
                                                "value": t['utterance'][start:end].lower(),
                                                "start": start,
                                                "end": end
                                            })
                                    else:
                                        # have slot & value
                                        if ontology['domains'][domain]['slots'][slot]['is_categorical']:
                                            for value in value_list:
                                                value = value.lower()
                                                if value not in ontology['domains'][domain]['slots'][slot]['possible_values'] and value != 'dontcare':
                                                    ontology['domains'][domain]['slots'][slot]['possible_values'].append(value)
                                                    print('add value to ontology', domain, slot, value)
                                                assert value in ontology['domains'][domain]['slots'][slot][
                                                    'possible_values'] or value == 'dontcare'
                                                turn['dialogue_act']['categorical'].append({
                                                    "intent": intent,
                                                    "domain": domain,
                                                    "slot": slot,
                                                    "value": value,
                                                })
                                        elif slot in numerical_slots:
                                            value = value_list[-1]
                                            (start, end), num = pharse_in_sen(value, t['utterance'])
                                            if num:
                                                assert value.lower() == t['utterance'][start:end].lower() \
                                                       or digit2word[value].lower() == t['utterance'][start:end].lower()
                                                turn['dialogue_act']['non-categorical'].append({
                                                    "intent": intent,
                                                    "domain": domain,
                                                    "slot": slot.lower(),
                                                    "value": t['utterance'][start:end].lower(),
                                                    "start": start,
                                                    "end": end
                                                })
                                        else:
                                            # span info in frame['slots']
                                            for value in value_list:
                                                for slot_info in frame['slots']:
                                                    start = slot_info['start']
                                                    end = slot_info['exclusive_end']
                                                    if slot_info['slot'] == slot and t['utterance'][start:end] == value:
                                                        turn['dialogue_act']['non-categorical'].append({
                                                            "intent": intent,
                                                            "domain": domain,
                                                            "slot": slot,
                                                            "value": value.lower(),
                                                            "start": start,
                                                            "end": end
                                                        })
                                                        break
                            # add span da to all_slot_spans_from_da
                            for ele in turn['dialogue_act']['non-categorical']:
                                all_slot_spans_from_da.insert(0, {
                                    "domain": ele["domain"],
                                    "slot": ele["slot"],
                                    "value": ele["value"].lower(),
                                    "utt_idx": utt_idx,
                                    "start": ele["start"],
                                    "end": ele["end"]
                                })
                            if speaker == 'user':
                                # DONE: record state update, may come from sys acts
                                # prev_state: state. update the state using current frames.
                                # candidate span info from prev frames and current frames
                                slot_spans = []
                                for frame in t['frames']:
                                    for ele in frame['slots']:
                                        slot, start, end = ele['slot'].lower(), ele['start'], ele['exclusive_end']
                                        slot_spans.append({
                                            "domain": service2domain(frame['service']),
                                            "slot": slot,
                                            "value": t['utterance'][start:end].lower(),
                                            "utt_idx": utt_idx,
                                            "start": start,
                                            "end": end
                                        })
                                for frame in prev_sys_frames:
                                    for ele in frame['slots']:
                                        slot, start, end = ele['slot'].lower(), ele['start'], ele['exclusive_end']
                                        slot_spans.append({
                                            "domain": service2domain(frame['service']),
                                            "slot": slot,
                                            "value": d['turns'][utt_idx-1]['utterance'][start:end].lower(),
                                            "utt_idx": utt_idx-1,
                                            "start": start,
                                            "end": end
                                        })
                                # turn['slot_spans'] = slot_spans
                                # turn['all_slot_span'] = deepcopy(all_slot_spans_from_da)
                                state_update = {"categorical": [], "non-categorical": []}
                                # print(utt_idx)
                                for frame in t['frames']:
                                    domain = service2domain(frame['service'])
                                    # print(domain)
                                    for slot, value_list in frame['state']['slot_values'].items():
                                        # For categorical slots, this list contains a single value assigned to the slot.
                                        # For non-categorical slots, all the values in this list are spoken variations
                                        # of each other and are equivalent (e.g, "6 pm", "six in the evening",
                                        # "evening at 6" etc.).
                                        numerical_equal_values = []
                                        if slot in numerical_slots:
                                            for value in value_list:
                                                if value in digit2word:
                                                    numerical_equal_values.append(digit2word[value])
                                        value_list += numerical_equal_values
                                        assert len(value_list) > 0, print(slot, value_list)
                                        assert slot in state[domain]
                                        value_list = list(set([x.lower() for x in value_list]))
                                        if state[domain][slot] in value_list:
                                            continue
                                        # new value
                                        candidate_values = value_list
                                        for prev_user_frame in prev_user_frames:
                                            prev_domain = service2domain(prev_user_frame['service'])
                                            if prev_domain == domain and slot in prev_user_frame['state']['slot_values']:
                                                prev_value_list = [x.lower() for x in prev_user_frame['state']['slot_values'][slot]]
                                                candidate_values = list(set(value_list) - set(prev_value_list))
                                        assert state[domain][slot] not in candidate_values
                                        assert candidate_values

                                        if ontology['domains'][domain]['slots'][slot]['is_categorical']:
                                            state_cnt.setdefault('cate_slot_update', 0)
                                            state_cnt['cate_slot_update'] += 1
                                            value = candidate_values[0]
                                            state_update['categorical'].append(
                                                {"domain": domain, "slot": slot, "value": value}
                                            )
                                            state[domain][slot] = value
                                        else:
                                            state_cnt.setdefault('non_cate_slot_update', 0)
                                            state_cnt['non_cate_slot_update'] += 1
                                            span_priority = []
                                            slot_spans_len = len(slot_spans)
                                            all_slot_spans = slot_spans+all_slot_spans_from_da
                                            for span_idx, slot_span in enumerate(all_slot_spans):
                                                priority = 0
                                                span_domain = slot_span['domain']
                                                span_slot = slot_span['slot']
                                                span_value = slot_span['value']
                                                if domain == span_domain:
                                                    priority += 1
                                                if slot == span_slot:
                                                    priority += 10
                                                if span_value in candidate_values:
                                                    priority += 100
                                                if span_idx + 1 <= slot_spans_len:
                                                    priority += 0.5
                                                span_priority.append(priority)
                                                if span_idx + 1 <= slot_spans_len:
                                                    # slot_spans not run out
                                                    if max(span_priority) >= 111.5:
                                                        break
                                                else:
                                                    # search in previous da
                                                    if max(span_priority) >= 111:
                                                        break
                                            if span_priority and max(span_priority) >= 100:
                                                # {111.5: 114255,
                                                #  111: 29591,
                                                #  100: 15208,
                                                #  110: 2159,
                                                #  100.5: 642,
                                                #  110.5: 125,
                                                #  101: 24}
                                                max_priority = max(span_priority)
                                                state_cnt.setdefault('max_priority', Counter())
                                                state_cnt['max_priority'][max_priority] += 1
                                                span_idx = np.argmax(span_priority)
                                                ele = all_slot_spans[span_idx]
                                                state_update['non-categorical'].append({
                                                    "domain": domain,
                                                    "slot": slot,
                                                    "value": ele['value'],
                                                    "utt_idx": ele["utt_idx"],
                                                    "start": ele["start"],
                                                    "end": ele["end"]
                                                })
                                                state[domain][slot] = ele['value']
                                            else:
                                                # not found
                                                value = candidate_values[0]
                                                state_update['non-categorical'].append(
                                                    {"domain": domain, "slot": slot, "value": value}
                                                )
                                                state[domain][slot] = value
                                                # print(t['utterance'])
                                                non_cate_slot_update_fail_cnt += 1
                                            non_cate_slot_update_cnt += 1
                                turn['state'] = deepcopy(state)
                                turn['state_update'] = state_update
                                prev_user_frames = deepcopy(t['frames'])
                            else:
                                prev_sys_frames = deepcopy(t['frames'])

                            for da in turn['dialogue_act']['binary']:
                                if da not in ontology['binary_dialogue_act']:
                                    ontology['binary_dialogue_act'].append(deepcopy(da))
                            dialogue['turns'].append(deepcopy(turn))
                        assert len(dialogue['turns']) % 2 == 0
                        dialogue['turns'].pop()
                        processed_dialogue.append(dialogue)
                        # break
        # sort ontology binary
        pprint(state_cnt)
        ontology['binary_dialogue_act'] = sorted(ontology['binary_dialogue_act'], key=lambda x:x['intent'])
        json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w'), indent=2)
        json.dump(processed_dialogue, open('data.json', 'w'), indent=2)
        write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        os.remove('data.json')
        print('# train dialog: {}, # train utterance: {}'.format(num_train_dialog, num_train_utt))
        print(non_cate_slot_update_fail_cnt, non_cate_slot_update_cnt) # 395 162399

    else:
        # read from file
        processed_dialogue = read_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        ontology = json.load(open(os.path.join(self_dir, 'ontology.json')))
    return processed_dialogue, ontology


if __name__ == '__main__':
    preprocess()
    print(match) # {'0': 4146, '1': 53626, '>1': 2904} =>(after user act released) {'0': 487, '1': 63886, '>1': 3097}
