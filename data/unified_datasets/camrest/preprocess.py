import zipfile
import json
import os
import copy
import logging

logging.basicConfig(level=logging.INFO)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(sys.path[-1])

from convlab2.util.file_util import read_zipped_json, write_zipped_json

self_dir = os.path.dirname(os.path.abspath(__file__))

cat_slot_values = {
    'area': ['north', 'east', 'west', 'south', 'centre'],
    'pricerange': ['cheap', 'moderate', 'expensive']
}

camrest_desc = {
    'restaurant': {
        'domain': 'find a restaurant to eat',
        'food': 'food type the restaurant serves',
        'area': 'area where the restaurant is located',
        'name': 'name of the restaurant',
        'pricerange': 'price range of the restaurant',
        'phone': 'phone number of the restaurant',
        'address': 'exact location of the restaurant',
        'postcode': 'postal code of the restaurant',
    },
    'intents': {
        'inform': 'inform user of value of a slot',
        'request': 'ask for value of a slot',
        'nooffer': 'inform user that no restaurant matches his request'
    }
}

all_slots = ['food', 'area', 'name', 'pricerange', 'phone', 'address', 'postcode']


def convert_da(utt, da, all_intent, all_binary_das):
    converted_da = {
        'binary': [],
        'categorical': [],
        'non-categorical': []
    }

    for _intent, svs in da.items():
        if _intent not in all_intent:
            all_intent.append(_intent)

        if _intent == 'nooffer':
            converted_da['binary'].append({
                'intent': _intent,
                'domain': 'restaurant',
                'slot': '',
                'value': ''
            })

            if {
                'intent': _intent,
                'domain': 'restaurant',
                'slot': '',
                'value': ''
            } not in all_binary_das:
                all_binary_das.append({
                    'intent': _intent,
                    'domain': 'restaurant',
                    'slot': '',
                    'value': ''
                })
            continue

        for s, v in svs:
            if 'care' in v:
                v = 'dontcare'
            s = s.lower()
            v = v.lower()
            if _intent == 'request':
                converted_da['binary'].append({
                    'intent': _intent,
                    'domain': 'restaurant',
                    'slot': s,
                    'value': ''
                })

                if {
                    'intent': _intent,
                    'domain': 'restaurant',
                    'slot': s,
                    'value': ''
                } not in all_binary_das:
                    all_binary_das.append({
                        'intent': _intent,
                        'domain': 'restaurant',
                        'slot': s,
                        'value': ''
                    })
                continue

            if s in cat_slot_values:
                assert v in cat_slot_values[s] + ['dontcare']
                converted_da['categorical'].append({
                    'intent': _intent,
                    'domain': 'restaurant',
                    'slot': s,
                    'value': v
                })

            else:
                # non-categorical
                start_ch = utt.find(v)

                if start_ch == -1:
                    # if not v == 'dontcare':
                    #     logging.info('non-categorical slot value not found')
                    #     logging.info('value: {}'.format(v))
                    #     logging.info('sentence: {}'.format(utt))
                    #     continue

                    converted_da['non-categorical'].append({
                        'intent': _intent,
                        'domain': 'restaurant',
                        'slot': s,
                        'value': v,
                        # 'start': 0,
                        # 'end': 0
                    })
                    continue

                converted_da['non-categorical'].append({
                    'intent': _intent,
                    'domain': 'restaurant',
                    'slot': s,
                    'value': utt[start_ch: start_ch + len(v)],
                    'start': start_ch,
                    'end': start_ch + len(v)
                })
                assert utt[start_ch: start_ch + len(v)] == v

    return converted_da


def convert_state(state, state_slots):
    ret_state = {'restaurant': {k: '' for k in state_slots}}
    for da in state:
        if da['act'] != 'inform':
            continue

        for s, v in da['slots']:
            s = s.lower()
            v = v.lower()

            if not s in all_slots:
                logging.info('state slot {} not in all_slots!'.format(s))
                continue

            ret_state['restaurant'][s] = v

            if s not in state_slots:
                print(s)
                raise

    return ret_state


def get_state_update(prev_state, cur_state, prev_turns, cur_user_da, dialog_id):
    # cur_user_da: List of non-categorical slot-values
    diff_state = {}
    state_update = {'categorical': [], 'non-categorical':[]}
    for s, v in cur_state.items():
        if s in prev_state and prev_state[s] == v:
            continue
        diff_state[s] = v

    for s, v in diff_state.items():
        if v == '':
            continue
        if s in cat_slot_values:
            assert v in cat_slot_values[s] + ['dontcare']
            state_update['categorical'].append({
                'domain': 'restaurant',
                'slot': s,
                'value': v,
            })
        else:
            # non-categorical slot
            found = False
            for _usr_da in cur_user_da:
                if _usr_da['slot'] == s and _usr_da['value'] == v :
                    found = True
                    if v != 'dontcare' and 'start' in _usr_da:
                        state_update['non-categorical'].append({
                            'domain': 'restaurant',
                            'slot': s,
                            'value': v,
                            'utt_idx': len(prev_turns),
                            'start': _usr_da['start'],
                            'end': _usr_da['end']
                        })
                    else:
                        state_update['non-categorical'].append({
                            'domain': 'restaurant',
                            'slot': s,
                            'value': v,
                        })
            if found:
                continue

            prev_sys_da = [] if len(prev_turns) == 0 else prev_turns[-1]['dialogue_act']['non-categorical']
            for _sys_da in prev_sys_da:
                if _sys_da['slot'] == s and _sys_da['value'] == v and 'start' in _sys_da:
                    if _sys_da['slot'] == s and _sys_da['value'] == v:
                        state_update['non-categorical'].append({
                            'domain': 'restaurant',
                            'slot': s,
                            'value': v,
                            'utt_idx': len(prev_turns) - 1,
                            'start': _sys_da['start'],
                            'end': _sys_da['end']
                        })
                        found = True

            if not found:
                state_update['non-categorical'].append({
                    'domain': 'restaurant',
                    'slot': s,
                    'value': v
                })

    return state_update


def preprocess():
    original_zipped_path = os.path.join(self_dir, 'original_data.zip')
    if not os.path.exists(original_zipped_path):
        raise FileNotFoundError(original_zipped_path)
    if not os.path.exists(os.path.join(self_dir, 'data.zip')) or not os.path.exists(
            os.path.join(self_dir, 'ontology.json')):
        # print('unzip to', new_dir)
        # print('This may take several minutes')
        archive = zipfile.ZipFile(original_zipped_path, 'r')
        archive.extractall(self_dir)

    all_data = []
    all_intent = []
    all_binary_das = []
    all_state_slots = ['pricerange', 'area', 'food']

    data_splits = ['train', 'val', 'test']
    extract_dir = os.path.join(self_dir, 'original_data')

    if not os.path.exists('data.zip') or not os.path.exists('ontology.json'):

        dialog_id = 1
        for data_split in data_splits:
            data = json.load(open(os.path.join(self_dir, extract_dir, '{}.json'.format(data_split))))

            for i, d in enumerate(data):

                dialogue = d['dial']
                converted_dialogue = {
                    'dataset': 'camrest',
                    'data_split': data_split,
                    'dialogue_id': 'camrest_' + str(dialog_id),
                    'original_id': d['dialogue_id'],
                    'domains': ['restaurant'],
                    'turns': []
                }

                prev_state = {'restaurant': {}}
                for turn in dialogue:
                    usr_text = turn['usr']['transcript'].lower()
                    usr_da = turn['usr']['dialog_act']

                    sys_text = turn['sys']['sent'].lower()
                    sys_da = turn['sys']['dialog_act']

                    cur_state = convert_state(turn['usr']['slu'], all_state_slots)
                    cur_user_da = convert_da(usr_text, usr_da, all_intent, all_binary_das)

                    usr_turn = {
                        'utt_idx': len(converted_dialogue['turns']),
                        'speaker': 'user',
                        'utterance': usr_text,
                        'dialogue_act': cur_user_da,
                        'state': copy.deepcopy(cur_state),
                        'state_update': get_state_update(prev_state['restaurant'], cur_state['restaurant'], converted_dialogue['turns'], cur_user_da['non-categorical'], converted_dialogue['dialogue_id'])
                    }

                    sys_turn = {
                        'utt_idx': len(converted_dialogue['turns'])+1,
                        'speaker': 'system',
                        'utterance': sys_text,
                        'dialogue_act': convert_da(sys_text, sys_da, all_intent, all_binary_das),
                    }

                    prev_state = copy.deepcopy(cur_state)

                    converted_dialogue['turns'].append(usr_turn)
                    converted_dialogue['turns'].append(sys_turn)
                if converted_dialogue['turns'][-1]['speaker'] == 'system':
                    converted_dialogue['turns'].pop(-1)
                all_data.append(converted_dialogue)
                dialog_id += 1

        json.dump(all_data, open('./data.json', 'w'), indent=4)
        write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        os.remove('data.json')

        new_ont = {
            'domains': {},
            'intents': {},
            'binary_dialogue_act': [],
            'state': {}
        }

        new_ont['state']['restaurant'] = {}
        for ss in all_state_slots:
            new_ont['state']['restaurant'][ss] = ''

        for b in all_binary_das:
            new_ont['binary_dialogue_act'].append(b)

        for i in all_intent:
            new_ont['intents'][i] = {'description': camrest_desc['intents'][i]}

        new_ont['domains']['restaurant'] = {
            'description': camrest_desc['restaurant']['domain'],
            'slots': {}
        }
        for s in all_slots:
            new_ont['domains']['restaurant']['slots'][s] = {
                "description": camrest_desc['restaurant'][s],
                "is_categorical": True if s in cat_slot_values else False,
                "possible_values": cat_slot_values[s] if s in cat_slot_values else []
            }
        json.dump(new_ont, open(os.path.join(self_dir, './ontology.json'), 'w'), indent=4)


    else:
        all_data = read_zipped_json(os.path.join(self_dir, './data.zip'), 'data.json')
        new_ont = json.load(open(os.path.join(self_dir, './ontology.json'), 'r'))

    return all_data, new_ont


if __name__ == '__main__':
    preprocess()
