import copy
import zipfile
import json
import os
from collections import Counter
from tqdm import tqdm
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.basicConfig(level=logging.INFO)
from convlab2.util.file_util import read_zipped_json, write_zipped_json

self_dir = os.path.dirname(os.path.abspath(__file__))

cat_slots = ['price range', 'area']
cat_slot_values = {
    'area': [
                        "east",
                        "west",
                        "center",
                        "north",
                        "south"
                    ],
    'price range': [
                        "expensive",
                        "moderate",
                        "dontcare",
                        "cheap"
                    ]
}

woz_desc = {
    'restaurant': {
        'domain': 'search for a restaurant to dine',
        'food': 'food type of the restaurant',
        'area': 'area of the restaurant',
        'postcode': 'postal code of the restaurant',
        'phone': 'phone number of the restaurant',
        'address': 'address of the restaurant',
        'price range': 'price range of the restaurant',
        'name': 'name of the restaurant'
    },
    'intents': {
        'inform': 'system informs user the value of a slot',
        'request': 'system asks the user to provide value of a slot',
    }
}


def convert_da(da, utt, all_binary):
    converted = {
        'binary': [],
        'categorical': [],
        'non-categorical': []
    }

    for s, v in da:
        v = 'expensive' if 'expensive' in v else v
        v = 'center' if v == 'centre' else v
        v = 'east' if 'east' in v else v

        if s in ['request']:
            _converted = {
                'intent': 'request',
                'domain': 'restaurant',
                'slot': v,
                'value': '',
            }
            converted['binary'].append(_converted)

            if _converted not in all_binary:
                all_binary.append(_converted)

        else:
            slot_name = s
            slot_type = 'categorical' if s in cat_slots else 'non-categorical'

            converted[slot_type].append({
                'intent': 'inform',
                'domain': 'restaurant',
                'slot': slot_name,
                'value': v
            })

            if slot_type == 'non-categorical':

                start = utt.find(v)

                if start != -1:
                    end = start + len(v)
                    converted[slot_type][-1]['start'] = start
                    converted[slot_type][-1]['end'] = end

    return converted


def convert_state(state):
    ret = {
        'restaurant': {}
    }
    for s in woz_desc['restaurant']:
        if s == 'domain':
            continue
        ret['restaurant'][s] = ''
    for s in state:
        assert s['act'] in ['request', 'inform']
        if s['act'] == 'inform':
            for _s, _v in s['slots']:
                _v = 'expensive' if 'expensive' in _v else _v
                _v = 'center' if _v == 'centre' else _v
                _v = 'east' if 'east' in _v else _v
                # try:
                # assert _s not in ret['restaurant']
                # except:
                #     continue
                ret['restaurant'][_s] = _v

    return ret


def get_state_update(prev_state, cur_state, usr_da, turn_idx, dialog_idx):

    ret = {
        'categorical': [],
        'non-categorical': []
    }
    for k, v in prev_state['restaurant'].items():

        if k in cur_state['restaurant'] and cur_state['restaurant'][k] == v:
            continue
        if k in cat_slots:
            ret['categorical'].append({
                'domain': 'restaurant',
                'slot': k,
                'value': cur_state['restaurant'][k]
            })
        else:
            found = False
            for _da in usr_da['non-categorical']:

                if _da['slot'] == k and _da['value'] == cur_state['restaurant'][k]:
                    found = True
                    if v == 'dontcare':
                        ret['non-categorical'].append({
                            'domain': 'restaurant',
                            'slot': k,
                            'value': cur_state['restaurant'][k],
                        })
                    else:
                        ret['non-categorical'].append({
                            'domain': 'restaurant',
                            'slot': k,
                            'value': cur_state['restaurant'][k]
                        })

                        if 'start' in _da:
                            ret['non-categorical'][-1].update({
                                'utt_idx': turn_idx * 2,
                                'start': _da['start'],
                                'end': _da['end']
                            })

            if not found:
                # print(dialog_idx, turn_idx*2)
                # print(k, v)
                # print('===================')
                ret['non-categorical'].append({
                    'domain': 'restaurant',
                    'slot': k,
                    'value': cur_state['restaurant'][k]
                })

    return ret



def preprocess():
    dataset_dir = 'woz'
    data_splits = ['train', 'validate', 'test']
    all_dialogues = []
    all_binary_intents = []
    all_slot = []
    all_slot_value = {}
    extract_dir = os.path.join(self_dir, 'original_data')

    if not os.path.exists('data.zip') or not os.path.exists('ontology.json'):
        # data not processed
        data_zip_file = os.path.join(self_dir, 'original_data.zip')
        if not os.path.exists(data_zip_file):
            raise FileNotFoundError(data_zip_file)

        logging.info('unzip woz data to {}'.format(extract_dir))
        archive = zipfile.ZipFile(data_zip_file, 'r')
        archive.extractall(extract_dir)

        dialog_id = 1
        for split in data_splits:

            data = json.load(open(os.path.join(self_dir, extract_dir, 'original_data/woz_{}_en.json'.format(split))))


            for dialogue in data:
                ret = {}
                ret['dataset'] = "woz"
                ret['data_split'] = split if split != 'validate' else 'val'
                ret['dialogue_id'] = 'woz_' + str(dialog_id)
                ret['original_id'] = split + str(dialogue['dialogue_idx']) if split != 'validate' else 'val' + str(dialogue['dialogue_idx'])
                ret['domains'] = ['restaurant']

                ret['turns'] = []

                turns = dialogue['dialogue']
                n_turn = len(turns)
                prev_state = {'restaurant':{k: '' for k in woz_desc['restaurant'] if k != 'domain'}}

                for i in range(n_turn):

                    sys_utt = turns[i]['system_transcript'].lower()
                    usr_utt = turns[i]['transcript'].lower()
                    usr_da = turns[i]['turn_label']
                    bs = turns[i]['belief_state']

                    for s, v in usr_da:
                        if s == 'request':
                            if v not in all_slot:
                                all_slot.append(v)

                            if v not in all_slot_value and v != 'dontcare':
                                all_slot_value[v] = []

                        else:
                            if s not in all_slot:
                                all_slot.append(s)
                            if v == 'dontcare':
                                continue
                            if s not in all_slot_value:
                                all_slot_value[s] = [v]
                            else:
                                if v not in all_slot_value[s]:
                                    all_slot_value[s].append(v)

                    if i != 0:
                        ret['turns'].append({
                            'utt_idx': len(ret['turns']),
                            'speaker': 'system',
                            'utterance': sys_utt,
                            'dialogue_act': {'binary':[], 'categorical': [], 'non-categorical':[]},
                        })

                    cur_state = convert_state(bs)
                    cur_usr_da = convert_da(usr_da, usr_utt, all_binary_intents)

                    ret['turns'].append({
                        'utt_idx': len(ret['turns']),
                        'speaker': 'user',
                        'utterance': usr_utt,
                        'state': cur_state,
                        'dialogue_act': cur_usr_da,
                        'state_update': get_state_update(prev_state, cur_state, cur_usr_da, i, ret['dialogue_id'])
                    })

                    prev_state = copy.deepcopy(cur_state)

                all_dialogues.append(ret)
                dialog_id += 1

        save_file = 'data.json'
        json.dump(all_dialogues, open(save_file, 'w'), indent=4)
        write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        os.remove('data.json')

        new_ont = {'domains': {
            'restaurant': {
                'description': woz_desc['restaurant']['domain'],
                'slots': {}
            }
        }, 'intents': {
            'inform': {
                'description': woz_desc['intents']['inform'],
            },
            'request': {
                'description': woz_desc['intents']['request'],
            },
        }, 'binary_dialogue_act': []
        }
        for i in all_binary_intents:
            new_ont['binary_dialogue_act'].append(i)

        for slot in all_slot_value:
            if slot in cat_slots:
                new_ont['domains']['restaurant']['slots'][slot] = {
                    'is_categorical': True,
                    'possible_values': [],
                    'description': woz_desc['restaurant'][slot]
                }
                for v in all_slot_value[slot]:
                    v = 'expensive' if 'expensive' in v else v
                    v = 'center' if v == 'centre' else v
                    v = 'east' if 'east' in v else v
                    if v not in new_ont['domains']['restaurant']['slots'][slot]['possible_values']:
                        new_ont['domains']['restaurant']['slots'][slot]['possible_values'].append(v)
            else:
                new_ont['domains']['restaurant']['slots'][slot] = {
                    'is_categorical': False,
                    'possible_values': [],
                    'description': woz_desc['restaurant'][slot]
                }

        new_ont['state'] = {
            'restaurant': {k: '' for k in all_slot_value}
        }

        json.dump(new_ont, open(os.path.join(self_dir, 'ontology.json'), 'w'), indent=4)

    else:
        # read from file
        all_dialogues = read_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        new_ont = json.load(open(os.path.join(self_dir, 'ontology.json')))

    return all_dialogues, new_ont


if __name__ == '__main__':
    preprocess()
