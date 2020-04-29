"""
extract all value appear in dialog act and state, for translation.
"""
import json
import zipfile
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def extract_value(data, all_da, all_state, all_value):
    for _, sess in data.items():
        for i, turn in enumerate(sess['log']):
            for domain_intent, svs in turn['dialog_act'].items():
                all_da.setdefault(domain_intent, {})
                domain = domain_intent.split('-')[0]
                all_value.setdefault(domain.lower(), {})
                for slot, value in svs:
                    all_da[domain_intent].setdefault(slot, {})
                    if domain!='general':
                        all_value[domain.lower()].setdefault(REF_SYS_DA[domain].get(slot, slot), {})
                    if '|' in value:
                        for sv in value.split('|'):
                            all_da[domain_intent][slot].setdefault(sv, 0)
                            all_da[domain_intent][slot][sv] += 1
                            all_value[domain.lower()][REF_SYS_DA[domain].get(slot, slot)][sv.lower()] = ''
                        continue
                    all_da[domain_intent][slot].setdefault(value, 0)
                    all_da[domain_intent][slot][value] += 1
                    if domain != 'general':
                        all_value[domain.lower()][REF_SYS_DA[domain].get(slot, slot)][value.lower()] = ''
            for domain in turn['metadata']:
                all_state.setdefault(domain, {})
                all_value.setdefault(domain, {})
                for slot, value in turn['metadata'][domain]['book'].items():
                    if slot == 'booked':
                        for book_info in turn['metadata'][domain]['book']['booked']:
                            for s, v in book_info.items():
                                all_state[domain].setdefault(s, {})
                                all_value[domain].setdefault(s, {})
                                if '|' in v:
                                    for sv in value.split('|'):
                                        all_state[domain][s].setdefault(sv, 0)
                                        all_state[domain][s][sv] += 1
                                        all_value[domain][s][sv.lower()] = ''
                                    continue
                                all_state[domain][s].setdefault(v, 0)
                                all_state[domain][s][v] += 1
                                all_value[domain][s][v.lower()] = ''
                    else:
                        all_state[domain].setdefault(slot, {})
                        all_value[domain].setdefault(slot, {})
                        if '|' in value:
                            for sv in value.split('|'):
                                all_state[domain][slot].setdefault(sv, 0)
                                all_state[domain][slot][sv] += 1
                                all_value[domain][slot][sv.lower()] = ''
                            continue
                        all_state[domain][slot].setdefault(value, 0)
                        all_state[domain][slot][value] += 1
                        all_value[domain][slot][value.lower()] = ''
                for slot, value in turn['metadata'][domain]['semi'].items():
                    all_state[domain].setdefault(slot, {})
                    all_value[domain].setdefault(slot, {})
                    if '|' in value:
                        for sv in value.split('|'):
                            all_state[domain][slot].setdefault(sv, 0)
                            all_state[domain][slot][sv] += 1
                            all_value[domain][slot][sv.lower()] = ''
                        continue
                    all_state[domain][slot].setdefault(value, 0)
                    all_state[domain][slot][value] += 1
                    all_value[domain][slot][value.lower()] = ''

if __name__ == '__main__':
    all_value = {}
    all_da = {}
    all_state = {}
    for s in ['train', 'val', 'test']:
        data = read_zipped_json(s+'.json.zip', s+'.json')
        extract_value(data, all_da, all_state, all_value)
    slotNotTranslate = ['stay', 'day', 'people', 'reference', 'postcode', 'ref', 'null', 'phone', 'leaveat', 'arriveby',
                        'taxi_phone', 'none', 'time', 'trainid', 'duration', 'ticket', 'entrance fee', 'price', 'choice']
    value2translate = {}
    for domain in all_value:
        for slot, value in all_value[domain].items():
            if not slot:
                print(value)
                continue
            slot = slot.lower()
            if slot not in slotNotTranslate:
                if slot in ['destination', 'departure']:
                    slot = 'name'
                if slot == 'taxi_types':
                    slot = 'type'
                value2translate.setdefault(slot.lower(), {})
                value2translate[slot.lower()].update(value)
            # else:
            #     print(slot)
            #     print(value[:20])
            #     print()
    json.dump({
        'all_da': all_da,
        'all_state': all_state,
        'all_value': all_value,
    }, open('all_value.json', 'w'), indent=2)
    json.dump(value2translate, open('value2translate.json', 'w'), indent=2)
    print('all value', len(all_value))
