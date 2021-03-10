import copy
import re
import zipfile
import json
import os
from tqdm import tqdm
import sys
import difflib
from fuzzywuzzy import fuzz
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from convlab2.util.file_util import read_zipped_json, write_zipped_json
import logging


logging.basicConfig(level=logging.INFO)
self_dir = (os.path.abspath(os.getcwd()))

REF_SYS_DA = {
    'Attraction': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Fee': "entrance fee", 'Name': "name", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Type': "type",
        'none': None, 'Open': None
    },
    'Hospital': {
        'Department': 'department', 'Addr': 'address', 'Post': 'postcode',
        'Phone': 'phone', 'none': None
    },
    'Booking': {
        'Day': 'day', 'Name': 'name', 'People': 'people',
        'Ref': 'Ref', 'Stay': 'stay', 'Time': 'time',
        'none': None
    },
    'Hotel': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Internet': "internet", 'Name': "name", 'Parking': "parking",
        'Phone': "phone", 'Post': "postcode", 'Price': "pricerange",
        'Stars': "stars", 'Type': "type", 'Stay': 'stay', 'Day': 'day', 'People': 'people',
        'none': None
    },
    'Restaurant': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Name': "name", 'Food': "food", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange",
        'Time': 'time', 'Day': 'day', 'People': 'people',
        'none': None
    },
    'Taxi': {
        'Arrive': "arriveBy", 'Car': "taxi_types", 'Depart': "departure",
        'Dest': "destination", 'Leave': "leaveAt", 'Phone': "taxi_phone",
        'none': None
    },
    'Train': {
        'Arrive': "arriveBy", 'Choice': "choice", 'Day': "day",
        'Depart': "departure", 'Dest': "destination",
        'Leave': "leaveAt", 'People': "people", 'Ref': "Ref",
        'Time': "duration", 'none': None, 'Ticket': 'price',
    },
    'Police': {
        'Addr': "address", 'Post': "postcode", 'Phone': "phone", 'none': None
    },
}

# taxi restaurant attraction train
slot_to_type = {
    'taxi-destination': 'non',
    'taxi-departure': 'non',
    'taxi-leaveAt': 'non',
    'taxi-arriveBy': 'non',
    'restaurant-food': 'non',
    'restaurant-name': 'non',
    'attraction-address': 'non',
    'attraction-postcode': 'non',
    'restaurant-pricerange': 'cat',
    'restaurant-address': 'non',
    'restaurant-area': 'cat',
    'restaurant-postcode': 'non',
    'attraction-phone': 'non',
    'attraction-entrance fee': 'non',
    'booking-time': 'non',
    'booking-day': 'cat',
    'attraction-type': 'non',
    'attraction-area': 'cat',
    'train-destination': 'non',
    'train-arriveBy': 'non',
    'train-departure': 'non',
    'hotel-internet': 'cat',
    'hotel-area': 'cat',
    'booking-stay': 'non',
    'booking-people': 'non',
    'train-duration': 'non',
    'train-people': 'non',
    'train-day': 'cat',
    'train-Ref': 'non',
    'hotel-stars': 'cat',
    'train-leaveAt': 'non',
    'train-price': 'non',
    'hotel-parking': 'cat',
    'hotel-phone': 'non',
    'hotel-name': 'non',
    'hotel-pricerange': 'cat',
    'hotel-people': 'non',
    'restaurant-phone': 'non',
    'hotel-postcode': 'non',
    'hotel-address': 'non',
    'attraction-name': 'non',
    'hotel-type': 'non',
    'restaurant-people': 'non',
    'train-choice': 'non',
    'attraction-pricerange': 'cat',
    'hotel-stay': 'non',
    'booking-name': 'non',
    'booking-Ref': 'non',
    'restaurant-time': 'non',
    'restaurant-day': 'cat',
    'hotel-day': 'cat',
    'hotel-choice': 'non',
    'restaurant-choice': 'non',
    'attraction-choice': 'non',
    'taxi-taxi_phone': 'non',
    'taxi-taxi_types': 'non',
    'police-address': 'non',
    'police-postcode': 'non',
    'police-phone': 'non'
}

state_cat_slot_value_dict = {
    "hotel-pricerange": {
        "cheap": 735,
        "moderate": 1063,
        "expensive": 594,
    },
    "hotel-parking": {
        "yes": 1809,
        "no": 126,
        "free": 4,
    },
    "hotel-day": {
        "tuesday": 385,
        "wednesday": 410,
        "monday": 365,
        "saturday": 407,
        "friday": 393,
        "thursday": 384,
        "sunday": 369,
    },
    "train-day": {
        "wednesday": 533,
        "monday": 533,
        "saturday": 543,
        "thursday": 547,
        "friday": 563,
        "tuesday": 553,
        "sunday": 613,
    },
    "hotel-stars": {
        "4": 1263,
        "2": 193,
        "0": 201,
        "3": 401,
        "5": 45,
        "1": 45,
    },
    "hotel-internet": {
        "yes": 1841,
        "no": 79,
        "free": 2
    },
    "hotel-area": {
        "east": 416,
        "north": 717,
        "centre": 538,
        "south": 289,
        "west": 316,
    },
    "attraction-area": {
        "centre": 1290,
        "west": 332,
        "north": 155,
        "south": 240,
        "east": 272,
    },
    "restaurant-pricerange": {
        "expensive": 1477,
        "cheap": 758,
        "moderate": 1028,
    },
    "restaurant-area": {
        "centre": 1745,
        "south": 398,
        "north": 390,
        "east": 360,
        "west": 423,
    },
    "restaurant-day": {
        "thursday": 362,
        "wednesday": 412,
        "friday": 395,
        "monday": 383,
        "sunday": 399,
        "saturday": 421,
        "tuesday": 350,
    }
}


synonyms = [
    ["el shaddia guesthouse", "el shaddai"],
    [ "peterborough", "peterbourgh"],
    ["night club", "nightclub", 'nightclubs'],
    ["boat", "boating"],
    ["portugese", "portuguese"],
    ["guesthouse", "guest house"],
    ["seafood", "sea food"],
    ["christ 's college", "christ college"],
    ["huntingdon marriott hotel"]
]

state_cat_slot_ds = [k for k, v in slot_to_type.items() if v == 'cat']

da_cat_slot_values = {
    # 'hotel-stay': ['1', '2', '3', '4', '5'],
    'hotel-internet': ['free', 'no', 'none', 'yes'],
    'hotel-parking': ['free', 'no', 'none', 'yes']
}

state_cat_slot_values = {}

multiwoz_desc = {
    'taxi': {
        'domain': 'taxi information query system',
        'taxi_phone': 'taxi phone number',
        'taxi_types': 'taxi type',
    },
    'restaurant': {
        'domain': 'restaurant information query system',
        'address': 'exact location of the restaurant',
        'postcode': 'postcode of the restaurant',
        'phone': 'restaurant phone number',
        'choice': 'number of restaurants meeting requests of user',
    },
    'attraction': {
        'domain': 'an entertainment that is offered to the public',
        'address': 'details of where the attraction is',
        'postcode': 'postcode of the attraction',
        'phone': 'phone number of the attraction',
        'entrance fee': 'the fee charged for admission to the attraction',
        'pricerange': 'the price range for the attraction, from cheap to expensive',
        'choice': 'number of attractions matching requests of user'
    },
    'booking': {
        'domain': 'to arrange with a taxi, restaurant, train, etc.',
        'time': 'time for an order',
        'day': 'day for an order, from monday to sunday',
        'stay': 'for how long the user wish to be at a place',
        'people': 'how many person the order is for',
        'name': 'name of the ordered place',
        'Ref': 'reference number of the order'
    },
    'train': {
        'domain': 'query and order a train',
        'duration': 'the length of time the train trip lasts',
        'Ref': 'reference number of the order',
        'price': 'price for the train ticket',
        'choice': 'number of trains that meets requests of the user',
    },
    'hotel': {
        'domain': 'to query hotel information and place an order',
        'address': 'exact location of the hotel',
        'postcode': 'postcode of the hotel',
        'phone': 'hotel phone number',
        'choice': 'number of hotels that meets requests of the user',
    },
    'police': {
        'domain': 'find police stations',
        'address': 'exact location of the police station',
        'postcode': 'postcode of the police station',
        'phone': 'police station phone number',
    },
    'intents': {
        'inform': 'inform user of value for a certain slot',
        'request': 'ask for value of a slot',
        'nobook': 'inform user of booking failure',
        'reqmore': 'ask user for more instructions',
        'book': 'place an order for user',
        'bye': 'end a conversation and say goodbye to user',
        'thank': 'express gratitude',
        'welcome': 'welcome',
        'offerbooked': 'inform user that an order is succussful',
        'recommend': 'recommend a choice for user request',
        'greet': 'express greeting',
        'nooffer': 'inform user that no options matches user request',
        'offerbook': 'offer to place an order for user',
        'select': 'provide several choices for user to choose from',
    }
}

digit2word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
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

    if phrase.isdigit() and phrase in digit2word:
        phrase = digit2word[phrase]
        p = re.compile(pw.format(re.escape(phrase)), re.I)
        m = re.search(p, sen)
        if m:
            num = len(re.findall(p, sen))
            # if num > 1:
            #     match['>1'] += 1
            # else:
            #     match['1'] += 1
            return m.span('v'), num
    # match['0'] += 1
    if phrase.isdigit():
        pattern = pn
    else:
        pattern = pw
    p = re.compile(pattern.format(re.escape(phrase)), re.I)
    m = re.search(p, sen)
    if m:
        num = len(re.findall(p, sen))
        # if num > 1:
        #     match['>1'] += 1
        # else:
        #     match['1'] += 1
        return m.span('v'), num
    return (None, None), 0





def update_state(state, update):
    # print('======================')
    # print(state)
    # print(update)
    # print('======================')

    for service, service_update in update.items():
        if service not in state:
            state[service] = copy.deepcopy(service_update)
        else:
            state[service].update(update[service])


def convert_da(utt, da_dict, binary_ont, intent_ont, did, tid, da_cat_slot_values):
    '''
     convert multiwoz dialogue acts to required format
    :param utt: user or system utt
    :param da_dict: multiwoz da
    :param binary_ont: binary ontology
    :param intent_ont: intent ontology
    :return:
    '''
    converted_da = {
        'categorical': [],
        'non-categorical': [],
        'binary': []
    }

    for Domain_Act, S, v in da_dict:
        Domain, Act = Domain_Act.split('-')
        if Domain.lower() in ['police', 'hospital', 'bus']:
            continue

        if Act.lower() not in intent_ont:
            intent_ont[Act.lower()] = {}

        # general domain is converted to empty domain. e.g. thank, bye
        if Domain == 'general':
            assert S == 'none'
            assert v == 'none'
            converted_dict = {
                'intent': Act.lower(),
                'domain': '',
                'slot': '',
                'value': ''
            }
            converted_da['binary'].append(converted_dict)

            if converted_dict not in binary_ont:
                binary_ont.append(converted_dict)
            continue



        try:
            reformated_slot = REF_SYS_DA[Domain][S]
        except:
            # print('44444444444444444444444444444444')
            # print(Domain, S)
            # logging.info('slot not in REF_SYS_DA, drop')
            continue

        # if slot is None, da should be converted into binary
        if reformated_slot is None:
            if not (S == 'none' and v == 'none'):
                # mainly for `Open` slot
                # print('11111111111111111111')
                # print(Domain_Act, S, v)
                continue
            # Booking-Inform none none
            # Police-Inform none none
            # Train-OfferBook none none
            converted_dict = {
                'intent': Act.lower(),
                'domain': Domain.lower(),
                'slot': '',
                'value': ''
            }
            converted_da['binary'].append(converted_dict)
            if converted_dict not in binary_ont:
                binary_ont.append(converted_dict)
            continue

        reformated_domain_slot = Domain.lower() + '-' + reformated_slot

        if Act.lower() == 'request':
            converted_dict = {
                'intent': 'request',
                'domain': Domain.lower(),
                'slot': reformated_slot,
                'value': ''
            }
            converted_da['binary'].append(converted_dict)

            if converted_dict not in binary_ont:
                binary_ont.append(converted_dict)
            continue

        # vs = da_dict[(Domain_Act, S)]['values']

        if reformated_domain_slot in slot_to_type and slot_to_type[reformated_domain_slot] == 'cat':
            origin_v = v
            v = v.lower()
            # if reformated_domain_slot in cat_slot_proj:
            #     v = cat_slot_proj[reformated_domain_slot][v]
            if reformated_domain_slot not in da_cat_slot_values:
                da_cat_slot_values[reformated_domain_slot] = []
            # if v not in cat_slot_values[reformated_domain_slot]:
            da_cat_slot_values[reformated_domain_slot].append(v)
            converted_da['categorical'].append({
                'intent': Act.lower(),
                'domain': Domain.lower(),
                'slot': reformated_slot,
                'value': v
            })
            if 'start_word' in da_dict[(Domain_Act, S, origin_v)]:
                start_ws = da_dict[(Domain_Act, S, origin_v)]['start_word']
                end_ws = da_dict[(Domain_Act, S, origin_v)]['end_word']
                utt_list = utt.split()
                for start_w, end_w in zip(start_ws, end_ws):
                    if start_w > len(utt_list) or end_w > len(utt_list):
                        continue
                    start_ch = 0
                    for i in range(start_w):
                        start_ch += len(utt_list[i]) + 1
                    end_ch = start_ch
                    for i in range(start_w, end_w):
                        end_ch += len(utt_list[i]) + 1
                    try:
                        end_ch += len(utt_list[end_w])
                    except:
                        print(utt_list, start_w, end_w)
                    if not utt[start_ch: end_ch] == origin_v:
                        # print('2222222222222222222222222')
                        # print('\n'.join([v, utt[start_ch: end_ch - 1]]))
                        continue

                    else:
                        converted_da['categorical'][-1].update({
                            'start': start_ch,
                            'end': end_ch
                        })
                        break;

        else:
            if 'start_word' not in da_dict[(Domain_Act, S, v)]:
                # todo no span annotation
                converted_da['non-categorical'].append({
                    'intent': Act.lower(),
                    'domain': Domain.lower(),
                    'slot': reformated_slot,
                    'value': v
                })
                continue

            start_ws = da_dict[(Domain_Act, S, v)]['start_word']
            end_ws = da_dict[(Domain_Act, S, v)]['end_word']
            utt_list = utt.split()
            found = True
            for start_w, end_w in zip(start_ws, end_ws):
                if start_w > len(utt_list) or end_w > len(utt_list):
                    continue
                start_ch = 0
                for i in range(start_w):
                    start_ch += len(utt_list[i]) + 1
                end_ch = start_ch
                for i in range(start_w, end_w):
                    end_ch += len(utt_list[i]) + 1
                try:
                    end_ch += len(utt_list[end_w])
                except:
                    print(utt_list, start_w, end_w)
                if not utt[start_ch: end_ch] == v:
                    # print('2222222222222222222222222')
                    # print('\n'.join([v, utt[start_ch: end_ch - 1]]))
                    continue

                else:
                    found = True
                    converted_da['non-categorical'].append({
                        'intent': Act.lower(),
                        'domain': Domain.lower(),
                        'slot': reformated_slot,
                        'value': v,
                        'start': start_ch,
                        'end': end_ch
                    })
                    break

            if not found:
                converted_da['non-categorical'].append({
                    'intent': Act.lower(),
                    'domain': Domain.lower(),
                    'slot': reformated_slot,
                    'value': v
                })
    return converted_da


def get_state_update(prev_state, cur_state, dialog, did, tid, utt, coref_dict, slot_notfound_dict, state_cat_slot_values):
    prev_turns = dialog['turns']
    state_update = {'categorical': [], 'non-categorical': []}
    notfoundnum = 0
    total_value = 0

    diff_state = {}
    if prev_state is None:
        diff_state = {domain: {slot: value for slot, value in cur_state[domain].items() if value != ''} for domain in
                      cur_state}
    else:
        assert len(prev_state) == len(cur_state), print(prev_state, cur_state)
        for domain, domain_state in prev_state.items():
            if domain not in diff_state:
                diff_state[domain] = {}
            for slot, value in domain_state.items():
                if value != cur_state[domain][slot]:
                    # assert len(cur_state[domain][slot]) > 0, print(did, tid, domain, slot, utt)
                    diff_state[domain][slot] = cur_state[domain][slot]

    ret_diff_state = copy.deepcopy(diff_state)



    for domain in diff_state:
        for slot in diff_state[domain]:

            total_value += 1
            fix_or = False
            if '|' in diff_state[domain][slot]:
                value = diff_state[domain][slot].split('|')[0]
            else:
                value = diff_state[domain][slot]

            # if dialog['original_id'] == 'PMUL2512' and tid == 17 and value == '02:45':
            #     value = '2:45'

            value_list = [value]
            for _synonyms in synonyms:
                if value in _synonyms:
                    value_list = _synonyms

            value_list.extend(get_time_variants(value))
            value_list.extend(get_genitive_variants(value))
            value_list.extend(get_bb_variants(value))

            if value.endswith(' restaurant'):
                value_list.append(value.split(' restaurant')[0])
            if value.endswith(' hotel'):
                value_list.append(value.split(' hotel')[0])
            found = False
            for value in value_list:
                # categorical slots
                if slot in ['internet', 'parking', 'pricerange', 'day', 'area', 'stars']:
                    reformated_domain_slot = '-'.join([domain, slot])
                    if reformated_domain_slot in state_cat_slot_value_dict and (value in state_cat_slot_value_dict[reformated_domain_slot] or value in ['dontcare', '', 'none', 'not mentioned']):
                        state_update['categorical'].append({
                            'domain': domain,
                            'slot': slot,
                            'value': diff_state[domain][slot]
                        })
                        if value != diff_state[domain][slot]:
                            state_update['categorical'][-1].update({'fixed_value': value})
                            ret_diff_state[domain][slot] = value
                        else :
                            for _turn in prev_turns[::-1]:
                                found = False
                                for da in _turn['dialogue_act']['categorical']:
                                    if da['value'] == value:
                                        if 'start' in da:
                                            state_update['categorical'][-1].update({
                                                'utt_idx': _turn['utt_idx'],
                                                'start': da['start'],
                                                'end': da['end'],
                                                'from': 'prev_da_span'
                                            })
                                            found = True
                                            break
                                if found:
                                    break
                    else:
                        state_update['categorical'].append({
                            'domain': domain,
                            'slot': slot,
                            'value': diff_state[domain][slot],
                            'fixed_value': 'not found'
                        })
                        ret_diff_state[domain][slot] = 'not found'
                        notfoundnum += 1
                    # reformated_domain_slot = '-'.join([domain, slot]
                    found = True
                    break

                # process value ---> none
                assert value not in ['none', 'not mentioned']
                if value in ['', 'dontcare']:
                    # if reformated_domain_slot not in state_cat_slot_values:
                    #     state_cat_slot_values[reformated_domain_slot] = []
                    # # if v not in cat_slot_values[reformated_domain_slot]:
                    # state_cat_slot_values[reformated_domain_slot].append(value)
                    state_update['non-categorical'].append({
                        'domain': domain,
                        'slot': slot,
                        'value': diff_state[domain][slot]
                    })
                    found = True
                    break

                # first look for values in coref_dict
                for _Domain_Act, _Slot, _value in coref_dict:
                    _domain, _act = _Domain_Act.lower().split('-')
                    _slot = _Slot.lower()
                    _coref_value = coref_dict[(_Domain_Act, _Slot, _value)]['coref_value']
                    if _coref_value == '':
                        continue
                    _coref_turn = coref_dict[(_Domain_Act, _Slot, _value)]['turn']
                    if _coref_turn == -1:
                        continue
                    _coref_pos = coref_dict[(_Domain_Act, _Slot, _value)]['pos']
                    if _coref_pos == '':
                        continue
                    _utt = coref_dict[(_Domain_Act, _Slot, _value)]['utt']
                    if _domain == domain and _slot == slot and value == _coref_value:

                        start_w, end_w = [int(p) for p in _coref_pos.split('-')]
                        utt_list = _utt.split()
                        start_ch = 0
                        for i in range(start_w):
                            start_ch += len(utt_list[i]) + 1
                        end_ch = start_ch
                        for i in range(start_w, end_w + 1):
                            end_ch += len(utt_list[i]) + 1
                        end_ch -= 1

                        if not _utt[start_ch: end_ch] == _coref_value:
                            # print(111111111111111111111111111111111)
                            # print(_utt[start_ch: end_ch], _coref_value)
                            continue

                        state_update['non-categorical'].append({
                            'domain': domain,
                            'slot': slot,
                            'value': diff_state[domain][slot],
                            'from': 'coref',
                            'utt_idx': _coref_turn,
                            'start': start_ch,
                            'end': end_ch
                        })
                        if value != diff_state[domain][slot]:
                            state_update['categorical'][-1].update({'fixed_value': value})
                            ret_diff_state[domain][slot] = value
                        found = True

                if found:
                    break

                # from da annotation
                for _turn in prev_turns[::-1]:
                    for da in _turn['dialogue_act']['non-categorical']:
                        # if da['domain'] == domain and da['slot'] == slot and fuzz.ratio(da['value'], value) > 85:
                            # if not da['value'] == value:
                            #     print(1111111111111111)
                            #     print(value, da['value'])

                        if fuzz.ratio(da['value'], value) > 85:

                            if 'start' in da:
                                found = True
                                state_update['non-categorical'].append({
                                    'domain': domain,
                                    'slot': slot,
                                    # 'value': da['value'],
                                    'value': diff_state[domain][slot],
                                    'utt_idx': _turn['utt_idx'],
                                    'start': da['start'],
                                    'end': da['end'],
                                    'from': 'prev_da_span'
                                })
                                if value != diff_state[domain][slot]:
                                    state_update['non-categorical'][-1].update({'fixed_value': value})
                                    ret_diff_state[domain][slot] = value
                                if da['value'] != value:
                                    state_update['non-categorical'][-1].update({'fixed_value':da['value']})
                                    ret_diff_state[domain][slot] = da['value']

                                break
                    if found:
                        break

                if found:
                    break

                # from utterance
                for _turn in prev_turns[::-1]:
                    _utt = _turn['utterance']
                    (start, end), num = pharse_in_sen(str(value), _utt)
                    if num:
                        assert value.lower() == _utt[start:end].lower() \
                               or digit2word[value].lower() == _utt[start:end].lower()
                        found = True
                        state_update['non-categorical'].append({
                            'domain': domain,
                            'slot': slot,
                            'value': diff_state[domain][slot],
                            # 'value': _utt[start:end].lower(),
                            # 'fixed_value': _utt[start:end].lower(),
                            'from': 'prev_utt',
                            'utt_idx': _turn['utt_idx'],
                            'start': start,
                            'end': end
                        })
                        if value != diff_state[domain][slot]:
                            state_update['non-categorical'][-1].update({'fixed_value': value})
                            ret_diff_state[domain][slot] = value
                        if value != _utt[start:end].lower():
                            state_update['non-categorical'][-1].update({'fixed_value': _utt[start:end].lower()})
                            ret_diff_state[domain][slot] = _utt[start:end].lower()
                        found = True
                        break
                if found:
                    break

                # from utterance
                if not value.isdigit():
                    for _turn in prev_turns[::-1]:
                        _utt = _turn['utterance']

                        s = difflib.SequenceMatcher(None, _utt, value)
                        matches = s.get_matching_blocks()

                        for i, j, n in matches:
                            possible_value = _utt[i: i+len(value)]

                            if i+ len(value) < len(_utt) and _utt[i+len(value)] not in [ ' ', ',', '.', '?', '!', '/'] :
                                possible_value += _utt[i+len(value):].split()[0]

                                if possible_value.startswith('th '):
                                    possible_value = possible_value[3:]
                                    i += 3
                            if i > 0 and _utt[i-1] not in [ ' ', ',', '.', '?', '!', '/']:
                                # cut first incomplete word
                                if len(possible_value.split()) > 1:
                                    i += len(possible_value.split()[0]) + 1
                                    possible_value = ' '.join(possible_value.split()[1:])


                                # prepend first incomplete word
                                # possible_value = _utt[:i].split()[-1] + possible_value
                                # i -= len(_utt[:i].split()[-1])


                            if fuzz.token_sort_ratio(value, possible_value) > 92 or possible_value.startswith('ashley hotel and lovell lodge') :
                                found = True

                                state_update['non-categorical'].append({
                                            'domain': domain,
                                            'slot': slot,
                                            'value': diff_state[domain][slot],
                                            # 'value': possible_value,
                                            # 'fixed_value': possible_value,
                                            'from':'prev_utt',
                                            'utt_idx': _turn['utt_idx'],
                                            'start': i,
                                            'end': i+len(possible_value)
                                        })
                                if value != diff_state[domain][slot]:
                                    state_update['non-categorical'][-1].update({'fixed_value': value})
                                    ret_diff_state[domain][slot] = value
                                if possible_value != value:
                                    state_update['non-categorical'][-1].update({'fixed_value': possible_value})
                                    ret_diff_state[domain][slot] = possible_value
                                break
                    #             assert _utt[i:i+len(possible_value)] == possible_value, print(_utt, _utt[i:i+len(possible_value)], possible_value)
                    #             break
                                # if not possible_value == value:
                                #             print(3333333333333333)
                                #             print(value)
                                #             print(possible_value)
                            if found:
                                break
                        if found:
                            break

                if found:
                    break
            if not found:
                #                 print('3333333333333333333')
                #                 print(did, tid)
                #                 print(domain, slot, value)
                #                 print([_t['utterance'] for _t in prev_turns])
                # assert slot not in ['internet', 'parking', 'pricerange', 'day', 'area', 'stars']

                if (domain, slot) not in slot_notfound_dict:
                    slot_notfound_dict[(domain, slot)] = 1
                else:
                    slot_notfound_dict[(domain, slot)] += 1
                state_update['non-categorical'].append({
                    'domain': domain,
                    'slot': slot,
                    'value': diff_state[domain][slot],
                    'fixed_value': 'not found'
                })
                ret_diff_state[domain][slot] = 'not found'
                notfoundnum += 1
    return state_update, notfoundnum, total_value, ret_diff_state


def merge_data_annotation():
    extract_dir = os.path.join(self_dir, 'original_data')
    data25 = json.load(open(os.path.join(self_dir, extract_dir, 'data_meta_fixed.json')))
    data21_train = json.load(open(os.path.join(self_dir, extract_dir, 'train.json')))
    data21_val = json.load(open(os.path.join(self_dir, extract_dir, 'val.json')))
    data21_test = json.load(open(os.path.join(self_dir, extract_dir, 'test.json')))
    data21 = {}
    data21.update(data21_train)
    data21.update(data21_val)
    data21.update(data21_test)

    update_from_25_cnt = 0
    total_turn = 0
    for dial_id, dialog in data21.items():
        dial_id = dial_id + '.json'
        assert dial_id in data25
        for i, _turn in enumerate(dialog['log']):
            total_turn += 1
            if _turn['text'] == data25[dial_id]['log'][i]['text']:
                _turn['span_info'].extend(copy.deepcopy(data25[dial_id]['log'][i]['span_info']))
                # _turn['span_info'] = list(set(_turn['span_info']))
                # _turn['dialog_act'].update(copy.deepcopy(data25[dial_id]['log'][i]['dialog_act']))
                for Domain_Intent in data25[dial_id]['log'][i]['dialog_act']:
                    if Domain_Intent in _turn['dialog_act']:
                        _turn['dialog_act'][Domain_Intent].extend(data25[dial_id]['log'][i]['dialog_act'][Domain_Intent])
                    else:
                        _turn['dialog_act'][Domain_Intent] = copy.deepcopy(data25[dial_id]['log'][i]['dialog_act'][Domain_Intent])
                    # _turn['dialog_act'][Domain_Intent] = list(set(_turn['dialog_act'][Domain_Intent]))
                if 'coreference' in data25[dial_id]['log'][i]:
                    _turn['coreference'] = copy.deepcopy(data25[dial_id]['log'][i]['coreference'])
                update_from_25_cnt += 1
            else:
                # print('==============multiwoz21=================')
                # print(_turn['text'])
                # print('==============multiwoz25=================')
                # print(data25[dial_id]['log'][i]['text'])
                continue

    print('{}/{} turns update from multiwoz25 data'.format(update_from_25_cnt, total_turn))
    return data21


def preprocess(da_cat_slot_values, state_cat_slot_values):
    all_data = []
    binary_ont = []
    intent_ont = {}
    state_ont = {}

    data_splits = ['train', 'val', 'test']
    # data_splits = ['test']
    extract_dir = os.path.join(self_dir, 'original_data')
    num_train_dialogue = 0
    num_train_utt = 0

    num_match_error_da_span = 0

    if not os.path.exists('data.zip') or not os.path.exists('ontology.json'):
        # for data_split in data_splits:
        data_zip_file = os.path.join(self_dir, 'original_data.zip')
        if not os.path.exists(data_zip_file):
            raise FileNotFoundError(data_zip_file)

        logging.info('unzip multiwoz data to {}'.format(extract_dir))
        archive = zipfile.ZipFile(data_zip_file, 'r')
        archive.extractall(extract_dir)

        data = merge_data_annotation()
        # exit()
        # data = json.load(open(os.path.join(self_dir, extract_dir, 'data_meta_fixed.json')))
        train_list = open(os.path.join(self_dir, extract_dir, 'trainListFile')).read().split()
        val_list = open(os.path.join(self_dir, extract_dir, 'valListFile')).read().split()
        test_list = open(os.path.join(self_dir, extract_dir, 'testListFile')).read().split()

        total_not_found_slot = 0
        total_slot = 0
        total_turn = 0
        total_not_found_turn = 0
        total_not_found_state = 0

        slot_notfound_dict = {}

        dialog_idx = 0
        for dialog_id, dialog in tqdm(data.items()):

            acc_not_found_flag = False

            coref_dict = {}

            data_split = None
            for _split in data_splits:
                if dialog_id.strip('.json') in eval(_split + '_list'):
                    data_split = _split
                    break
            # assert data_split is not None
            # if data_split != 'test':
            #     continue

            if data_split == 'train':
                num_train_dialogue += len(data)

            dialog_idx += 1
            # if dialog_idx > 10:
            #     break
            converted_dialogue = {
                'dataset': 'multiwoz21',
                'data_split': data_split,
                'dialogue_id': 'multiwoz21_' + str(dialog_idx),
                'original_id': dialog_id,
                'domains': [d for d in dialog['goal'] if
                            len(dialog['goal'][d]) != 0 and d in multiwoz_desc and d not in ['police', 'hospital', 'bus']],
                'turns': [],
            }

            if data_split == 'train':
                num_train_utt += len(dialog['log'])

            prev_state = None
            accum_fixed_state = {}
            for turn_id, turn in enumerate(dialog['log']):

                utt = turn['text'].lower()
                # for several wrong words
                utt = utt.replace('seeuni', 'see uni')

                utt = ' '.join(utt.split())
                das = turn['dialog_act']
                role = 'user' if turn_id % 2 == 0 else 'system'
                spans = turn['span_info']

                da_dict = {}
                for Domain_Act in das:
                    Domain = Domain_Act.split('-')[0]
                    if Domain.lower() not in converted_dialogue['domains'] and Domain.lower() not in ['general', 'booking']:
                        continue

                    Svs = das[Domain_Act]
                    for S, v in Svs:
                        v = v.lower()
                        if v.startswith('th '):
                            # print(v)
                            v = v[3:]
                        if v.startswith('he '):
                            # print(v)
                            v = v[3:]

                        if (Domain_Act, S, v) not in da_dict:
                            da_dict[(Domain_Act, S, v)] = {}

                for span in spans:
                    Domain_Act, S, v, start_word, end_word = span
                    v = v.lower()
                    if not (Domain_Act, S, v) in da_dict:
                        # logging.info('span da annotation not found in multiwoz da label')
                        # logging.info(dialog_id, turn_id)
                        # logging.info((Domain_Act, S, v))
                        # logging.info(da_dict)
                        num_match_error_da_span += 1
                    else:
                        if v.startswith('th '):
                            # print(v)
                            v = v[3:]
                            start_word += 3
                        if v.startswith('he '):
                            # print(v)
                            v = v[3:]
                            start_word += 3

                        if 'start_word' not in da_dict[(Domain_Act, S, v)]:
                            da_dict[(Domain_Act, S, v)]['start_word'] = []
                            da_dict[(Domain_Act, S, v)]['end_word'] = []

                        da_dict[(Domain_Act, S, v)]['start_word'].append(start_word)
                        da_dict[(Domain_Act, S, v)]['end_word'].append(end_word)

                converted_turn = {
                    'utt_idx': turn_id,
                    'speaker': role,
                    'utterance': utt,
                    'dialogue_act': convert_da(utt, da_dict, binary_ont, intent_ont, dialog_id, turn_id, da_cat_slot_values),
                }

                # for state annotations
                if role == 'system':
                    turn_state = turn['metadata']
                    cur_state = {}
                    for domain in turn_state:
                        if domain in ['police', 'hospital', 'bus']:
                            continue
                        if domain not in converted_dialogue['domains']:
                            continue
                        cur_state[domain] = {}
                        for subdomain in ['semi', 'book']:
                            for slot in turn_state[domain][subdomain]:
                                if slot == 'booked':
                                    continue
                                if slot == 'ticket':  # or (domain == 'train' and slot == 'people'):
                                    # for cases where domain slot exists in REF but not in state
                                    # because of check in evaluate.py
                                    continue

                                else:
                                    fixed_slot = slot
                                state_ds = domain + '-' + fixed_slot
                                if state_ds not in slot_to_type:
                                    logging.info('state slot not defined in da list')
                                    logging.info(state_ds)
                                if turn_state[domain][subdomain][slot] in ['', [], 'not mentioned', 'none']:
                                    cur_state[domain][fixed_slot] = ""
                                else:
                                    if turn_state[domain][subdomain][slot].startswith('th '):
                                        # print('state')
                                        # print(turn_state[domain][subdomain][slot])
                                        turn_state[domain][subdomain][slot] = turn_state[domain][subdomain][slot][3:]
                                    if turn_state[domain][subdomain][slot].startswith('he '):
                                        # print('state')
                                        # print(turn_state[domain][subdomain][slot])
                                        turn_state[domain][subdomain][slot] = turn_state[domain][subdomain][slot][3:]

                                    cur_state[domain][fixed_slot] = turn_state[domain][subdomain][slot]

                                if domain not in state_ont:
                                    state_ont[domain] = []
                                if fixed_slot not in state_ont[domain]:
                                    state_ont[domain].append(fixed_slot)

                        if domain == 'train' and 'people' not in cur_state[domain]:
                            cur_state[domain]['people'] = ''
                        # if len(converted_turn['state'][domain]) == 0:
                        #     converted_turn['state'].pop(domain)
                        if len(converted_dialogue['turns']) > 0:
                            # move state from system side to user side
                            converted_dialogue['turns'][-1]['state'] = copy.deepcopy(cur_state)

                    # for state update annotations
                    state_update, _notfoundslot, _totalslot, ret_diff_state = get_state_update(prev_state, cur_state, converted_dialogue,
                                                                               dialog_id, turn_id, turn['text'], coref_dict,
                                                                               slot_notfound_dict, state_cat_slot_values)

                    update_state(accum_fixed_state, ret_diff_state)
                    for domain in accum_fixed_state:
                        for slot in accum_fixed_state[domain]:
                            assert isinstance(accum_fixed_state[domain][slot], str), print(accum_fixed_state[domain][slot])

                    if _notfoundslot == 0:
                        # for slot in state_update['categorical']:
                        #     assert 'fixed_value' not in slot
                        for slot in state_update['non-categorical']:
                            if slot['value'] not in ['', 'dontcare']:
                                assert 'utt_idx' in slot

                    else:
                        flag = False
                        for slot in state_update['categorical']:
                            if 'fixed_value' in slot:
                                flag = True
                                break
                        for slot in state_update['non-categorical']:
                            if 'utt_idx' not in slot:
                                flag = True
                                break
                        assert flag, print(flag, state_update['non-categorical'])

                    total_turn += 1
                    total_slot += _totalslot
                    total_not_found_slot += _notfoundslot
                    total_not_found_turn += 1 if _notfoundslot > 0 else 0
                    if _notfoundslot > 0:
                        acc_not_found_flag = True
                    if acc_not_found_flag:
                        total_not_found_state += 1

                    coref_dict = {}
                    converted_dialogue['turns'][-1]['state_update'] = copy.deepcopy(state_update)
                    converted_dialogue['turns'][-1]['fixed_state'] = copy.deepcopy(accum_fixed_state)
                    if 'state' not in converted_dialogue['turns'][-1]:
                        converted_dialogue['turns'][-1]['state'] = {}
                    prev_state = copy.deepcopy(cur_state)

                converted_dialogue['turns'].append(converted_turn)

                if 'coreference' in turn:
                    for Domain_Act in turn['coreference']:
                        for Slot, value, coref, coref_turn, coref_pos in turn['coreference'][Domain_Act]:
                            value = value.lower()
                            coref_dict[(Domain_Act, Slot, value)] = {'turn': coref_turn, 'pos': coref_pos,
                                                                     'coref_value': coref,
                                                                     'utt': converted_dialogue['turns'][coref_turn][
                                                                         'utterance']}

            check_spans(converted_dialogue)
            postprocess_update_spans(converted_dialogue)
            if converted_dialogue['turns'][-1]['speaker'] == 'system':
                converted_dialogue['turns'].pop(-1)
            all_data.append(converted_dialogue)

        print('total_turn', total_turn)
        print('total_not_found_turn', total_not_found_turn)
        print('total_slot', total_slot)
        print('total_not_found_slot', total_not_found_slot)
        print('total_not_found_state', total_not_found_state)
        print(slot_notfound_dict)
        from collections import Counter
        # print({k : dict(Counter(v)) for k, v in cat_slot_values.items()})
        json.dump({k : dict(Counter(v)) for k, v in state_cat_slot_values.items()}, open(os.path.join(self_dir, 'cat_slot_values.json'), 'w'), indent=4)
        cat_slot_values = {k: list(set(v)) for k, v in state_cat_slot_values.items()}
        da_cat_slot_values = {k: list(set(v)) for k, v in da_cat_slot_values.items()}

        json.dump(all_data, open('data.json', 'w'), indent=4)
        write_zipped_json(os.path.join(self_dir, './data.zip'), 'data.json')
        os.remove('data.json')

        new_ont = {
            'domains': {},
            'intents': {},
            'binary_dialogue_act': {}
        }

        for d_s in slot_to_type:
            d, s = d_s.split('-')
            if d not in new_ont['domains']:
                new_ont['domains'][d] = {
                    'description': multiwoz_desc[d]['domain'],
                    'slots': {}
                }
            domain_ont = new_ont['domains'][d]
            assert s not in domain_ont
            domain_ont['slots'][s] = {
                'description': multiwoz_desc[d][s] if s in multiwoz_desc[d] else '',
                'is_categorical': d_s in state_cat_slot_ds,
                'possible_values': da_cat_slot_values[d_s] if d_s in state_cat_slot_ds else []
            }

        new_ont['state'] = {}
        # print(state_cat_slot_value_dict)
        print(state_ont)
        for d in state_ont:
            new_ont['state'][d] = {}
            for s in state_ont[d]:
                d_s = '-'.join([d, s])
                new_ont['state'][d][s] = {
                    'description': multiwoz_desc[d][s] if s in multiwoz_desc[d] else '',
                    'is_categorical': d_s in state_cat_slot_value_dict,
                    'possible_values': list(state_cat_slot_value_dict[d_s].keys()) if d_s in state_cat_slot_value_dict else []
                }

        new_ont['intents'] = {i: {'description': multiwoz_desc['intents'][i]} for i in intent_ont}
        new_ont['binary_dialogue_act'] = binary_ont

        slot_desc = json.load(open(os.path.join(self_dir, extract_dir, './slot_descriptions.json')))
        for domain_slot in slot_desc:
            _domain, _slot = domain_slot.split('-')
            _desc = slot_desc[domain_slot][0]
            if _slot == 'arriveby':
                _slot = 'arriveBy'
            elif _slot == 'leaveat':
                _slot = 'leaveAt'
            if 'book' in _slot:
                _slot = _slot.replace('book ', '')
            if not _domain in new_ont['state']:
                # logging.info('domain {} not in state domains'.format(_domain))
                continue
            if _domain in new_ont['domains'] and _slot in new_ont['domains'][_domain]['slots']:
                new_ont['domains'][_domain]['slots'][_slot]['description'] = _desc
            if not _slot in new_ont['state'][_domain]:
                logging.info('domain {} slot {} not in state'.format(_domain, _slot))
                continue
            # new_ont['state'][_domain][_slot] = ""
            assert _domain in new_ont['domains'], print(_domain)
            assert _slot in new_ont['domains'][_domain]['slots']

        logging.info('num_match_error_da_span {}'.format(num_match_error_da_span))
        json.dump(new_ont, open(os.path.join(self_dir, './ontology.json'), 'w'), indent=4)

    else:
        all_data = read_zipped_json(os.path.join(self_dir, './data.zip'), 'data.json')
        new_ont = json.load(open(os.path.join(self_dir, './ontology.json'), 'r'))
    logging.info('# dialogue: {}, # turn: {}'.format(num_train_dialogue, num_train_utt))
    return all_data, new_ont


def postprocess_update_spans(dialog):
    changed_utt_idx_and_position = {}
    for turn in dialog['turns']:
        if turn['speaker'] != 'user':
            continue
        changed = False
        for _update in turn['state_update']['non-categorical']:
            if 'utt_idx' in _update:
                utt_idx = _update['utt_idx']
                start = _update['start']
                end = _update['end']

                # assume at most one word changes for every utterance
                if turn['utt_idx'] not in changed_utt_idx_and_position:
                    if utt_idx == turn['utt_idx'] and start-1 > -1 and turn['utterance'][start-1] not in [' ']:
                        changed_utt_idx_and_position[turn['utt_idx']] = start
                        print('=======================')
                        print(dialog['original_id'])
                        print(turn['utterance'])
                        print(json.dumps(_update, indent=2))
                        print(turn['utterance'][start: end])
                        turn['utterance'] = turn['utterance'][:start] + ' ' + turn['utterance'][start:]
                        print(turn['utterance'])
                        _update['start'] += 1
                        _update['end'] += 1
                        changed = True
                if utt_idx not in changed_utt_idx_and_position:
                    continue
                else:
                    value = _update['fixed_value'] if 'fixed_value' in _update and _update['fixed_value'] != 'not found' else _update['value']
                    if start >= changed_utt_idx_and_position[utt_idx]:
                        if dialog['turns'][utt_idx]['utterance'][_update['start']: _update['end']] != value:
                            assert dialog['turns'][utt_idx]['utterance'][_update['start']+1: _update['end']+1] == value, print(dialog['turns'][utt_idx]['utterance'], dialog['turns'][utt_idx]['utterance'][_update['start']+1: _update['end']+1])
                            _update['start'] += 1
                            _update['end'] += 1
                    elif start < changed_utt_idx_and_position[utt_idx] < end:
                        if dialog['turns'][utt_idx]['utterance'][_update['start']: _update['end']] != value:
                            assert (dialog['turns'][utt_idx]['utterance'][_update['start']: _update['end']+1]).replace(' ', '') == value.replace(' ', ''), print(dialog['turns'][utt_idx]['utterance'], dialog['turns'][utt_idx]['utterance'][_update['start']: _update['end']+1], value)
                            print('fix')
                            print(_update)
                            _update['end'] += 1
                            _update['fixed_value'] = turn['utterance'][_update['start']: _update['end'] + 1].strip()
                            print(_update)
        if changed:
            for _update in turn['state_update']['non-categorical']:
                if 'utt_idx' in _update:
                    utt_idx = _update['utt_idx']
                    start = _update['start']
                    end = _update['end']

                    if utt_idx not in changed_utt_idx_and_position:
                        continue
                    else:
                        value = _update['fixed_value'] if 'fixed_value' in _update and _update[
                            'fixed_value'] != 'not found' else _update['value']
                        if start >= changed_utt_idx_and_position[utt_idx]:
                            if dialog['turns'][utt_idx]['utterance'][_update['start']: _update['end']] != value:
                                assert dialog['turns'][utt_idx]['utterance'][_update['start'] + 1: _update['end'] + 1] == value
                                _update['start'] += 1
                                _update['end'] += 1
                        elif start < changed_utt_idx_and_position[utt_idx] < end:
                            if dialog['turns'][utt_idx]['utterance'][_update['start']: _update['end']] != value:
                                print('====================fix===================')
                                print(_update)
                                assert (dialog['turns'][utt_idx]['utterance'][_update['start']: _update['end']+1]).replace(' ', '') == value.replace(' ', ''), print(dialog['turns'][utt_idx]['utterance'], dialog['turns'][utt_idx]['utterance'][_update['start']+1: _update['end']+1])
                                _update['end'] += 1
                                _update['fixed_value'] = dialog['turns'][utt_idx]['utterance'][_update['start']: _update['end'] + 1]
                                print(_update)
    for turn in dialog['turns']:
        if turn['speaker'] != 'user':
            continue
        for _update in turn['state_update']['non-categorical']:
            if 'utt_idx' in _update:
                value = _update['fixed_value'] if 'fixed_value' in _update and _update[
                    'fixed_value'] != 'not found' else _update['value']
                utt_idx = _update['utt_idx']
                start = _update['start']
                end = _update['end']
                if dialog['turns'][utt_idx]['utterance'][start] == ' ':
                    _update['start'] += 1
                    _update['fixed_value'] = value[1:]
                    value = value[1:]
                    start += 1
                assert dialog['turns'][utt_idx]['utterance'][start: end] == value, print(json.dumps(turn, indent=4), [c for c in dialog['turns'][utt_idx]['utterance'][start: end]], [c for c in value])
    return dialog


def get_time_variants(time_text):
    value_list = [time_text]
    pattern_time = r'(\d{1,2}:\d{2})(\s)?(am|pm|AM|PM)?'
    match_times = re.findall(pattern_time, time_text)
    if len(match_times) < 1:
        return []
    match_time = match_times[0]

    am_flag = match_time[2] in ['am', 'AM']
    pm_flag = match_time[2] in ['pm', 'PM']
    no_am_pm_flag = match_time[2] == ''
    if am_flag:
        # 4:00am -> 4:00
        value_list.append(match_time[0])
        if len(match_time[0]) == 4:
            # 4:00 -> 04:00
            value_list.append('0' + match_time[0])
    if pm_flag:
        # 4:00pm -> 16:00
        hour, min = match_time[0].split(':')
        hour = int(hour)
        new_hour = 12 + hour
        value_list.append(str(new_hour)+':'+min)
    if no_am_pm_flag:
        hour, min = match_time[0].split(':')
        hour = int(hour)
        if hour > 12:
            new_hour = hour - 12
            value_list.append(str(new_hour) + ':' + min + 'pm')
            value_list.append(str(new_hour) + ':' + min + ' pm')
            value_list.append(str(new_hour) + ':' + min)
            if min == '00':
                value_list.append(str(new_hour) + 'pm')
                value_list.append(str(new_hour) + ' pm')
                value_list.append(str(new_hour))
        else:
            value_list.append(str(hour) + ':' + min + 'am')
            value_list.append(str(hour) + ':' + min + ' am')
            value_list.append(str(hour) + ':' + min)
            if min == '00':
                value_list.append(str(hour) + 'am')
                value_list.append(str(hour) + ' am')
                value_list.append(str(hour))
        if len(match_time[0]) == 5 and match_time[0][0] == '0':
            value_list.append(match_time[0][1:])
        value_list.append(''.join(match_time[0].split(':')))

    return value_list


def get_genitive_variants(value):
    ret_list = []
    value_genitive_format = r"(?=\w)s(?=\s)"
    value_pattern = re.compile(value_genitive_format)

    span_genitive_value = re.sub(value_pattern, " 's", value)
    if span_genitive_value != value:
        ret_list.append(span_genitive_value)
    span_genitive_value = re.sub(value_pattern, "'s", value)
    if span_genitive_value != value:
        ret_list.append(span_genitive_value)
    # if len(ret_list) > 0:
    #     print('=============================')
    #     print(value)
    #     print(re.findall(value_pattern, value))
    #     print(ret_list)
    return ret_list


def check_spans(dialog):
    for turn in dialog['turns']:
        if turn['speaker'] != 'user':
            continue
        for _update in turn['state_update']['non-categorical']:
            if 'utt_idx' in _update:
                value = _update['fixed_value'] if 'fixed_value' in _update and _update[
                    'fixed_value'] != 'not found' else _update['value']
                utt_idx = _update['utt_idx']
                start = _update['start']
                end = _update['end']
                assert dialog['turns'][utt_idx]['utterance'][start:end] == value, print(dialog['turns'][utt_idx]['utterance'], dialog['turns'][utt_idx]['utterance'][start:end])



def get_bb_variants(value):
    ret_list = []
    if 'bed and breakfast' in value:
        ret_list.append(value.replace('bed and breakfast', 'b & b'))
    return ret_list

if __name__ == '__main__':
    preprocess(da_cat_slot_values, state_cat_slot_values)