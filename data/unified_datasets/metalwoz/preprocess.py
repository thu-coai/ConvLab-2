import json
import os
from zipfile import ZipFile, ZIP_DEFLATED

import json_lines


dataset = 'metalwoz'
self_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(self_dir)), 'data')
# origin_data_dir = os.path.join(DATA_PATH, dataset)
origin_data_dir = self_dir


def preprocess():
    ontology = {
        'domains': {},
        'intents': {},
        'binary_dialogue_act': [],
        'state': {}
    }

    def process_dialog(ori_dialog, split, dialog_id):
        domain = ori_dialog['domain']
        ontology['domains'][domain] = {
            'description': "",
            'slots': {}
        }
        dialog = {
            "dataset": dataset,
            "data_split": split,
            "dialogue_id": f'{dataset}_{dialog_id}',
            "original_id": ori_dialog['id'],
            "domains": [domain],
        }
        turns = []
        # starts with system
        for utt_idx, utt in enumerate(ori_dialog['turns'][1:]):
            turn = {
                'utt_idx': utt_idx,
                'utterance': utt,
                'dialogue_act': {
                    'categorical': [],
                    'non-categorical': [],
                    'binary': [],
                },
            }
            if utt_idx % 2 == 0:
                turn['speaker'] = 'user'
                turn['state'] = {}
                turn['state_update'] = {
                    'categorical': [],
                    'non-categorical': [],
                }
            else:
                turn['speaker'] = 'system'
            turns.append(turn)
        if turns[-1]['speaker'] == 'system':
            turns.pop()

        dialog['turns'] = turns
        return dialog

    dialog_id = 0
    data = []
    with ZipFile(os.path.join(origin_data_dir, 'metalwoz-v1.zip')) as zipfile:
        for path in zipfile.namelist():
            if path.startswith('dialogues'):
                for dialog in json_lines.reader(zipfile.open(path)):
                    data.append(process_dialog(dialog, 'train', dialog_id))
                    dialog_id += 1

    ZipFile(os.path.join(origin_data_dir, 'metalwoz-test-v1.zip')).extract('dstc8_metalwoz_heldout.zip')
    with ZipFile(os.path.join('dstc8_metalwoz_heldout.zip')) as zipfile:
        for path in zipfile.namelist():
            if path.startswith('dialogues'):
                for dialog in json_lines.reader(zipfile.open(path)):
                    data.append(process_dialog(dialog, 'test', dialog_id))
                    dialog_id += 1
    os.remove('dstc8_metalwoz_heldout.zip')

    json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w'))
    json.dump(data, open('data.json', 'w'), indent=4)
    ZipFile(os.path.join(self_dir, 'data.zip'), 'w', ZIP_DEFLATED).write('data.json')
    os.remove('data.json')


if __name__ == '__main__':
    preprocess()
