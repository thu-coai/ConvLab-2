"""
extract all value appear in dialog act and state, for translation.
"""
import json
import zipfile
import re
from pprint import pprint

def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
multi_name = set()

def extract_ontology(data):
    intent_set = set()
    domain_set = set()
    slot_set = set()
    value_set = {}
    for _, sess in data.items():
        for i, turn in enumerate(sess['messages']):
            for intent, domain, slot, value in turn['dialog_act']:
                intent_set.add(intent)
                domain_set.add(domain)
                slot_set.add(slot)
                if not domain in value_set:
                    value_set[domain] = {}
                elif slot not in value_set[domain]:
                    value_set[domain][slot] = set()
                elif slot in ['推荐菜', '名称', '酒店设施']:
                    if slot == '名称' and len(value.split()) > 1:
                        multi_name.add((domain, slot, value))
                    elif slot == '推荐菜' and '-' in value:
                        print((domain, slot, value))
                    for dish in value.split():
                        value_set[domain][slot].add(dish)
                elif slot == 'selectedResults' and domain in ['地铁', '出租']:
                    if domain == '地铁':
                        value = value[5:]
                        value_set[domain][slot].add(value.strip())
                    if domain == '出租':
                        value = value[4:-1]
                        for v in value.split(' - '):
                            value_set[domain][slot].add(v.strip())
                else:
                    value_set[domain][slot].add(value)
            if turn['role'] == 'usr':
                for _, domain, slot, value, __ in turn['user_state']:
                    domain_set.add(domain)
                    slot_set.add(slot)
                    if isinstance(value, list):
                        for v in value:
                            if not domain in value_set:
                                value_set[domain] = {}
                            elif slot not in value_set[domain]:
                                value_set[domain][slot] = set()
                            elif slot in ['推荐菜', '名称', '酒店设施']:
                                if slot == '名称' and len(value.split()) > 1:
                                    multi_name.add((domain, slot, value))
                                elif slot == '推荐菜' and '-' in value:
                                    print((domain, slot, value))
                                for dish in v.split():
                                    value_set[domain][slot].add(dish)
                            elif slot == 'selectedResults' and domain in ['地铁', '出租']:
                                if domain == '地铁':
                                    v = v[5:]
                                    value_set[domain][slot].add(v.strip())
                                if domain == '出租':
                                    v = v[4:-1]
                                    for item in v.split(' - '):
                                        value_set[domain][slot].add(item.strip())
                            else:
                                value_set[domain][slot].add(v)
                    else:
                        assert isinstance(value, str)
                        if not domain in value_set:
                            value_set[domain] = {}
                        elif slot not in value_set[domain]:
                            value_set[domain][slot] = set()
                        elif slot in ['推荐菜', '名称', '酒店设施']:
                            if slot == '名称' and len(value.split()) > 1:
                                multi_name.add((domain, slot, value))
                            elif slot == '推荐菜' and '-' in value:
                                print((domain, slot, value))
                            for dish in value.split():
                                value_set[domain][slot].add(dish)
                        elif slot == 'selectedResults' and domain in ['地铁', '出租']:
                            if domain == '地铁':
                                value = value[5:]
                                value_set[domain][slot].add(value.strip())
                            if domain == '出租':
                                value = value[4:-1]
                                for v in value.split(' - '):
                                    value_set[domain][slot].add(v.strip())
                        else:
                            value_set[domain][slot].add(value)
            else:
                for state_key in ['sys_state', 'sys_state_init']:
                    for domain, svd in turn[state_key].items():
                        domain_set.add(domain)
                        for slot, value in svd.items():
                            slot_set.add(slot)
                            if isinstance(value, list):
                                for v in value:
                                    if not domain in value_set:
                                        value_set[domain] = {}
                                    elif slot not in value_set[domain]:
                                        value_set[domain][slot] = set()
                                    elif slot in ['推荐菜', '名称', '酒店设施']:
                                        if slot == '名称' and len(value.split()) > 1:
                                            multi_name.add((domain, slot, value))
                                        elif slot == '推荐菜' and '-' in value:
                                            print((domain, slot, value))
                                        for dish in v.split():
                                            value_set[domain][slot].add(dish)
                                    elif slot == 'selectedResults' and domain in ['地铁', '出租']:
                                        if domain == '地铁':
                                            v = v[5:]
                                            value_set[domain][slot].add(v.strip())
                                        if domain == '出租':
                                            v = v[4:-1]
                                            for item in v.split(' - '):
                                                value_set[domain][slot].add(item.strip())
                                    else:
                                        value_set[domain][slot].add(v)
                            else:
                                assert isinstance(value, str)
                                if not domain in value_set:
                                    value_set[domain] = {}
                                elif slot not in value_set[domain]:
                                    value_set[domain][slot] = set()
                                elif slot in ['推荐菜', '名称', '酒店设施']:
                                    if slot == '名称' and len(value.split()) > 1:
                                        multi_name.add((domain, slot, value))
                                    elif slot == '推荐菜' and '-' in value:
                                        print((domain, slot, value))
                                    for dish in value.split():
                                        value_set[domain][slot].add(dish)
                                elif slot == 'selectedResults' and domain in ['地铁', '出租']:
                                    if domain == '地铁':
                                        value = value[5:]
                                        value_set[domain][slot].add(value.strip())
                                    if domain == '出租':
                                        value = value[4:-1]
                                        for v in value.split(' - '):
                                            value_set[domain][slot].add(v.strip())
                                else:
                                    value_set[domain][slot].add(value)
    return intent_set, domain_set, slot_set, value_set


if __name__ == '__main__':
    intent_set = set()
    domain_set = set()
    slot_set = set()
    value_set = {}

    for s in ['train', 'val', 'test', 'dstc9_data']:
        print(f'Proceeding {s} set...')
        data = read_zipped_json(s+'.json.zip', s+'.json')
        output = extract_ontology(data)
        intent_set |= output[0]
        domain_set |= output[1]
        slot_set |= output[2]
        for domain in output[3]:
            if domain in value_set:
                for slot in output[3][domain]:
                    if slot in value_set[domain]:
                        value_set[domain][slot] |= output[3][domain][slot]
                    else:
                        value_set[domain][slot] = output[3][domain][slot]
            else:
                value_set[domain] = output[3][domain]

        print(len(domain_set))
        print(len(intent_set))
        print(len(slot_set))
        print(len(value_set))

    intent_set = list(set([s.lower() for s in intent_set]))
    domain_set = list(set([s.lower() for s in domain_set]))
    slot_set = list(set([s.lower() for s in slot_set]))
    for domain in value_set:
        for slot in value_set[domain]:
            value_set[domain][slot] = list(set([s.lower() for s in value_set[domain][slot]]))
    # json.dump({
    #     'intent_set': intent_set,
    #     'domain_set': domain_set,
    #     'slot_set': slot_set,
    #     'value_set': value_set,
    # }, open('all_value_train_val_test.json', 'w'), indent=2, ensure_ascii=False)
    print(len(domain_set))
    print(len(intent_set))
    print(len(slot_set))
    print(len(value_set))
    print(len(multi_name))
    pprint(multi_name)

    # 统计一下每个domain-slot对应的value数
    # value_count = {x: len(value_set[x]) for x in value_set.keys()}
    # pprint(value_count)
