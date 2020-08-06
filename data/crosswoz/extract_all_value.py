"""
extract all value appear in dialog act and state, for translation.
"""
import json
import zipfile


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def extract_ontology(data):
    intent_set = set()
    domain_set = set()
    slot_set = set()
    value_set = set()
    for _, sess in data.items():
        for i, turn in enumerate(sess['messages']):
            for intent, domain, slot, value in turn['dialog_act']:
                intent_set.add(intent)
                domain_set.add(domain)
                slot_set.add(slot)
                value_set.add(value)
            if turn['role'] == 'usr':
                for _, domain, slot, value, __ in turn['user_state']:
                    domain_set.add(domain)
                    slot_set.add(slot)
                    if isinstance(value, list):
                        for v in value:
                            value_set.add(v)
                    else:
                        assert isinstance(value, str)
                        value_set.add(value)
            else:
                for domain, svd in turn['sys_state_init'].items():
                    domain_set.add(domain)
                    for slot, value in svd.items():
                        slot_set.add(slot)
                        if isinstance(value, list):
                            for v in value:
                                value_set.add(v)
                        else:
                            assert isinstance(value, str)
                            value_set.add(value)
    return intent_set, domain_set, slot_set, value_set


if __name__ == '__main__':
    intent_set = set()
    domain_set = set()
    slot_set = set()
    value_set = set()
    for s in ['train', 'val', 'test']:
        data = read_zipped_json(s+'.json.zip', s+'.json')
        output = extract_ontology(data)
        intent_set |= output[0]
        domain_set |= output[1]
        slot_set |= output[2]
        value_set |= output[3]
    intent_set = list(set([s.lower() for s in intent_set]))
    domain_set = list(set([s.lower() for s in domain_set]))
    slot_set = list(set([s.lower() for s in slot_set]))
    value_set = list(set([s.lower() for s in value_set]))
    json.dump({
        'intent_set': intent_set,
        'domain_set': domain_set,
        'slot_set': slot_set,
        'value_set': value_set,
    }, open('all_value.json', 'w'), indent=2, ensure_ascii=False)
    print(len(domain_set))
    print(len(intent_set))
    print(len(slot_set))
    print(len(value_set))
