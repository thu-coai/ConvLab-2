import json
import zipfile
from collections import Counter, OrderedDict

def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

def extract_act(data, counter):
    for dialogs in list(data.values()):
        for turn, meta in enumerate(dialogs['log']):
            if turn % 2 == 0:
                # usr turn
                continue
            action = ''
            if meta['dialog_act']:
                for act, slots in meta['dialog_act'].items():
                    for slot in slots:
                        action += f'{act}-'
                        if slot[1] == '?' or slot[0] == 'none':
                            action += f'{slot[0]}-{slot[1]};'
                        else:
                            action += f'{slot[0]}-1;'
                counter.update([action[:-1]])

if __name__ == '__main__':
    counter = Counter()
    for s in ['train', 'val', 'test']:
        data = read_zipped_json(s + '.json.zip', s + '.json')
        extract_act(data, counter)

    with open('da_slot_cnt.json', 'w') as f:
        json.dump(OrderedDict(counter.most_common()), f, indent=2)
