#!/usr/bin/env python
# coding: utf-8

# # Prepare 

import json
import re
from convlab2.util.multiwoz.multiwoz_slot_trans import *
en_pattern = re.compile('[a-zA-Z]')

vocab_dict_path = 'vocab_dict.json'
vocab_dict = json.load(open(vocab_dict_path))

# # MultiWOZ ontology translate


def capitalize(s):
    return s[0].upper() + s[1:]


def intent_translate(intent):
    return vocab_dict['intent_set'][intent.lower()]


def domain_translate(domain):
    return vocab_dict['domain_set'][domain.lower()]


def slot_translate(domain, slot):
    if slot == 'taxi_phone':
        slot = 'phone'
    elif slot == 'taxi_types':
        slot = 'car'

    try:
        if slot.lower() in vocab_dict['slot_set']:
            return vocab_dict['slot_set'][slot.lower()]
        elif capitalize(domain) in REF_USR_DA and slot.lower() in REF_USR_DA[capitalize(domain)]:
            return vocab_dict['slot_set'][REF_USR_DA[capitalize(domain)][slot.lower()].lower()]
        elif capitalize(domain) in REF_SYS_DA and capitalize(slot) in REF_SYS_DA[capitalize(domain)]:
            return vocab_dict['slot_set'][REF_SYS_DA[capitalize(domain)][capitalize(slot)].lower()]

    except:
        print(domain, slot)
        raise


def value_translate(domain, slot, value):
    value = value.strip().lower()
    error_tup = []
    if (domain, slot, value) in error_tup:
        print(f'    Skip mismatching tuple: {domain}-{slot}-{value}')
        return ''

    if domain.lower() == 'general':
        return value

    if domain not in vocab_dict['value_set']:
        domain = domain.lower()

    if slot not in vocab_dict['value_set'][domain]:
        slot = slot.lower()

    if slot not in vocab_dict['value_set'][domain] and slot.lower() in REF_USR_DA[capitalize(domain)]:
        slot = REF_USR_DA[capitalize(domain)][slot.lower()]

    if slot not in vocab_dict['value_set'][domain] and capitalize(slot) in REF_SYS_DA[capitalize(domain)]:
        slot = REF_SYS_DA[capitalize(domain)][capitalize(slot)]

    if slot not in vocab_dict['value_set'][domain]:
        if slot is None:
            slot = 'none'
    if slot not in vocab_dict['value_set'][domain]:
        print(f'{domain}-{slot} has no vocab dict')
        raise Exception

    if not en_pattern.findall(value):
        return value

    if not vocab_dict['value_set'][domain][slot]:
        return value

    if slot in ['post', 'postcode', 'phone', 'Ref', 'reference']:
        return value

    if value not in vocab_dict['value_set'][domain][slot]:
        if '|' in value:
            return '或'.join([value_translate(domain, slot, v) for v in value.split('|')])

        trans_value = {
            "king's college": "king 's college",
            "intermediate dependency area": "intermediate dependancy area"
            # "saint john's college": "saint johns college",
            # "kettle's yard": "kettles yard",
            # "christ's college": "christ college",
            # "little saint mary's church": "little saint marys church",
            # "people's portraits exhibition at girton college": "peoples portraits exhibition at girton college",
        }
        if value in trans_value:
            value = trans_value[value]
        elif value.replace("'", "") in vocab_dict['value_set'][domain][slot]:
            value = value.replace("'", "")
        elif value.replace("'s", "") in vocab_dict['value_set'][domain][slot]:
            value = value.replace("'s", "")
        elif value.replace('the ', '') in vocab_dict['value_set'][domain][slot]:
            value = value.replace('the ', '')

    try:
        return vocab_dict['value_set'][domain][slot][value]
    except:
        print(domain, slot, value)
        raise Exception


def ontology_translate(typ, *args):
    assert typ in ['intent', 'domain', 'slot', 'value'], 'Function translate() requires 1st argument: domain|slot|value.'
    if typ == 'intent':
        assert len(args) == 1, 'Needs 1 argument: intent.'
        return intent_translate(args[0])
    elif typ == 'domain':
        assert len(args) == 1, 'Needs 1 argument: domain.'
        return domain_translate(args[0])
    elif typ == 'slot':
        assert len(args) == 2, 'Needs 2 argument: domain, slot.'
        return slot_translate(args[0], args[1])
    elif typ == 'value':
        assert len(args) == 3, 'Needs 3 argument: domain, slot, value.'
        return value_translate(args[0], args[1], args[2])


if __name__ == '__main__':
    print(ontology_translate('intent', 'request'))
    print(ontology_translate('domain', 'police'))
    print(ontology_translate('slot', 'attraction', 'phone'))
    print(ontology_translate('value', 'attraction', 'area', 'central district'))
