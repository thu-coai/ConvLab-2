# -*- coding: utf-8 -*-
#! python3
import copy
import json
import os
import sys
import re
import shutil
import urllib
from urllib.request import urlopen
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
import difflib
import numpy as np

from convlab2.util.multiwoz.state import default_state

np.set_printoptions(precision=3)

np.random.seed(2)

'''
Most of the codes are from https://github.com/budzianowski/multiwoz
USING PYTHON 3.7
'''

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 50
IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']

fin = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mapping.pair'),'r')
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def normalize(text, clean_value=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    if clean_value:
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # normalize postcode
        ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                        text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    if clean_value:
        # replace time and and price
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [value_price] ', text)
        #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text) # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def fixDelex(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return data

    # if not isinstance(turn, str) and not isinstance(turn, unicode):
    if not isinstance(turn, str):
        for k, act in turn.items():
            if 'Attraction' in k:
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "attraction")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "attraction")
            if 'Hotel' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "hotel")
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "hotel")
            if 'Restaurant' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "restaurant")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "restaurant")

    return data


def getDialogueAct(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    acts = []
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return acts
    # for both of python2.7 and python3 on my pc, unicode is not found, tempopryli deleted
    # if not isinstance(turn, str) and not isinstance(turn, unicode):
    if not isinstance(turn, str):
        for k in turn.keys():
            # temp = [k.split('-')[0].lower(), k.split('-')[1].lower()]
            # for a in turn[k]:
            #     acts.append(temp + [a[0].lower()])

            if k.split('-')[1].lower() == 'request':
                for a in turn[k]:
                    acts.append(a[0].lower())
            elif k.split('-')[1].lower() == 'inform':
                for a in turn[k]:
                    acts.append([a[0].lower(), normalize(a[1].lower(), False)])

    return acts


def get_summary_bstate(bstate, get_domain=False):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi',u'restaurant',  u'hospital', u'hotel',u'attraction', u'train', u'police']
    summary_bstate = []
    summary_bvalue = []
    active_domain = []
    for domain in domains:
        domain_active = False

        booking = []
        #print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if len(bstate[domain]['book']['booked'])!=0:
                    booking.append(1)
                    # summary_bvalue.append("book {} {}:{}".format(domain, slot, "Yes"))
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                    summary_bvalue.append(["{}-book {}".format(domain, slot.strip().lower()), normalize(bstate[domain]['book'][slot].strip().lower(), False)]) #(["book", domain, slot, bstate[domain]['book'][slot]])
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                slot_enc[1] = 1
                summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()), "dontcare"]) #(["semi", domain, slot, "dontcare"])
            elif bstate[domain]['semi'][slot]:
                summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()), normalize(bstate[domain]['semi'][slot].strip().lower(), False)]) #(["semi", domain, slot, bstate[domain]['semi'][slot]])
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
            active_domain.append(domain)
        else:
            summary_bstate += [0]

    #print(len(summary_bstate))
    assert len(summary_bstate) == 94
    if get_domain:
        return active_domain
    else:
        return summary_bstate, summary_bvalue


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        #print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal'] if 'goal' in d else None  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    # last_bvs = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                print ('not ascii')
                return None
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                print ('not ascii')
                return None
            belief_summary, belief_value_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = str(belief_summary)
            d['log'][i]['belief_value_summary'] = belief_value_summary
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    sys_a = [t['dialogue_acts'] if 'dialogue_acts' in t else [] for t in d_orig['sys_log']]
    bvs = [t['belief_value_summary'] for t in d_orig['sys_log']]
    domain = [t['domain'] for t in d_orig['usr_log']]
    for item in zip(usr, sys, sys_a, domain, bvs):
        dial.append({'usr':item[0],'sys':item[1], 'sys_a':item[2], 'domain':item[3], 'bvs':item[4]})
    return dial


def loadData():
    data_url = "data/multi-woz/data.json"
    dataset_url = "https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y"
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/multi-woz")

    if not os.path.exists(data_url):
        print("Downloading and unzipping the MultiWOZ dataset")
        resp = urllib.request.urlopen(dataset_url)
        zip_ref = ZipFile(BytesIO(resp.read()))
        zip_ref.extractall("data/multi-woz")
        zip_ref.close()
        shutil.copy('data/multi-woz/MULTIWOZ2 2/data.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/valListFile.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/testListFile.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/dialogue_acts.json', 'data/multi-woz/')


def getDomain(idx, log, domains, last_domain):
    if idx == 1:
        active_domains = get_summary_bstate(log[idx]["metadata"], True) 
        crnt_doms = active_domains[0] if len(active_domains)!=0 else domains[0]
        return crnt_doms
    else:
        ds_diff = get_ds_diff(log[idx-2]["metadata"], log[idx]["metadata"])
        if len(ds_diff.keys()) == 0: # no clues from dialog states
            crnt_doms = last_domain
        else:
            crnt_doms = ds_diff.keys()
        # print("crnt_doms type : ", type(crnt_doms))
        crnt_doms = list(crnt_doms)
        return crnt_doms[0] # How about multiple domains in one sentence senario ?


def get_ds_diff(prev_d, crnt_d):
    diff = {}
    # Sometimes, metadata is an empty dictionary, bug?
    if not prev_d or not crnt_d:
        return diff

    for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
        assert k1 == k2
        if v1 != v2: # updated
            diff[k2] = v2
    return diff


def create_data(dialogue):
    data = {}
    domains = ["restaurant", "taxi", "police", "hospital", "hotel", "attraction", "train"]
    dialogue_name = "real-time"
    last_domain = ""
    for idx, turn in enumerate(dialogue['log']):
        origin_text = normalize(turn['text'], False)
        dialogue['log'][idx]['text'] = origin_text

        if idx % 2 == 1:  # if it's a system turn
            cur_domain = getDomain(idx, dialogue['log'], domains, last_domain)
            last_domain = [cur_domain]
            dialogue['log'][idx - 1]['domain'] = cur_domain
    data[dialogue_name] = dialogue
    for dialogue_name in data:
        # print dialogue_name
        dial = get_dial(data[dialogue_name])
        if dial:
            dialogue = {}
            dialogue['dialogue_idx'] = dialogue_name
            dialogue['domains'] = list(set([d['domain'] for d in dial]))
            last_bs = []
            dialogue['dialogue'] = []

            for turn_i, turn in enumerate(dial):
                # usr, usr_o, sys, sys_o, sys_a, domain
                turn_dialog = {}
                turn_dialog['system_transcript'] = dial[turn_i-1]['sys'] if turn_i > 0 else ""
                turn_dialog['turn_idx'] = turn_i
                turn_dialog['belief_state'] = [{"slots": [s], "act": "inform"} for s in turn['bvs']]
                turn_dialog['turn_label'] = [bs["slots"][0] for bs in turn_dialog['belief_state'] if bs not in last_bs]
                turn_dialog['transcript'] = turn['usr']
                turn_dialog['system_acts'] = dial[turn_i-1]['sys_a'] if turn_i > 0 else []
                turn_dialog['domain'] = turn['domain']
                last_bs = turn_dialog['belief_state']
                dialogue['dialogue'].append(turn_dialog)
    with open('data/real_dials.json', 'w') as f:
        json.dump([dialogue], f, indent=4)
    return [dialogue]


def createData():
    # download the data
    loadData()
    
    # create dictionary of delexicalied values that then we will search against, order matters here!
    # dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    fin1 = open('data/multi-woz/data.json', 'r')
    data = json.load(fin1)

    fin2 = open('data/multi-woz/dialogue_acts.json', 'r')
    data2 = json.load(fin2)

    for didx, dialogue_name in enumerate(data):

        dialogue = data[dialogue_name]

        domains = []
        for dom_k, dom_v in dialogue['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL: # check whether contains some goal entities
                domains.append(dom_k)

        idx_acts = 1
        last_domain, last_slot_fill = "", []
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            origin_text = normalize(turn['text'], False)
            # origin_text = delexicalize.markEntity(origin_text, dic)
            dialogue['log'][idx]['text'] = origin_text

            if idx % 2 == 1:  # if it's a system turn

                cur_domain = getDomain(idx, dialogue['log'], domains, last_domain)
                last_domain = [cur_domain]

                dialogue['log'][idx - 1]['domain'] = cur_domain
                dialogue['log'][idx]['dialogue_acts'] = getDialogueAct(dialogue_name, dialogue, data2, idx, idx_acts)
                idx_acts += 1

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)
        
        delex_data[dialogue_name] = dialogue

        # if didx > 10:
        #     break

    # with open('data/multi-woz/woz2like_data.json', 'w') as outfile:
    #     json.dump(delex_data, outfile)

    return delex_data


def buildDelexDict(origin_sent, delex_sent):
    dictionary = {}
    s = difflib.SequenceMatcher(None, delex_sent.split(), origin_sent.split())
    bs = s.get_matching_blocks()
    for i, b in enumerate(bs):
        if i < len(bs)-2:
            a_start = b.a + b.size
            b_start = b.b + b.size
            b_end = bs[i+1].b
            dictionary[a_start] = " ".join(origin_sent.split()[b_start:b_end])
    return dictionary


def divideData(data):
    """Given test and validation sets, divide
    the data for three different sets"""
    testListFile = []
    fin = open('data/multi-woz/testListFile.json', 'r')
    for line in fin:
        testListFile.append(line[:-1])
    fin.close()

    valListFile = []
    fin = open('data/multi-woz/valListFile.json', 'r')
    for line in fin:
        valListFile.append(line[:-1])
    fin.close()

    trainListFile = open('data/trainListFile', 'w')

    test_dials = []
    val_dials = []
    train_dials = []
        
    # dictionaries
    word_freqs_usr = OrderedDict()
    word_freqs_sys = OrderedDict()

    count_train, count_val, count_test = 0, 0, 0
    
    for dialogue_name in data:
        # print dialogue_name
        dial_item = data[dialogue_name]
        domains = []
        for dom_k, dom_v in dial_item['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL: # check whether contains some goal entities
                domains.append(dom_k)

        dial = get_dial(data[dialogue_name])
        if dial:
            dialogue = {}
            dialogue['dialogue_idx'] = dialogue_name
            dialogue['domains'] = list(set(domains)) #list(set([d['domain'] for d in dial]))
            last_bs = []
            dialogue['dialogue'] = []

            for turn_i, turn in enumerate(dial):
                # usr, usr_o, sys, sys_o, sys_a, domain
                turn_dialog = {}
                turn_dialog['system_transcript'] = dial[turn_i-1]['sys'] if turn_i > 0 else ""
                turn_dialog['turn_idx'] = turn_i
                turn_dialog['belief_state'] = [{"slots": [s], "act": "inform"} for s in turn['bvs']]
                turn_dialog['turn_label'] = [bs["slots"][0] for bs in turn_dialog['belief_state'] if bs not in last_bs] 
                turn_dialog['transcript'] = turn['usr']
                turn_dialog['system_acts'] = dial[turn_i-1]['sys_a'] if turn_i > 0 else []
                turn_dialog['domain'] = turn['domain']
                last_bs = turn_dialog['belief_state']
                dialogue['dialogue'].append(turn_dialog)
            
            if dialogue_name in testListFile:
                test_dials.append(dialogue)
                count_test += 1
            elif dialogue_name in valListFile:
                val_dials.append(dialogue)
                count_val += 1
            else:
                trainListFile.write(dialogue_name + '\n')
                train_dials.append(dialogue)
                count_train += 1

    print("# of dialogues: Train {}, Val {}, Test {}".format(count_train, count_val, count_test))

    # save all dialogues
    with open('data/dev_dials.json', 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open('data/test_dials.json', 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open('data/train_dials.json', 'w') as f:
        json.dump(train_dials, f, indent=4)

    # return word_freqs_usr, word_freqs_sys


def main():
    dialogue = {
        "log": [
            {
                "text": "am looking for a place to to stay that has cheap price range it should be in a type of hotel",
                "metadata": {}
            },
            {
                "text": "Okay, do you have a specific area you want to stay in?",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [],
                            "stay": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "not mentioned",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                }
            },
            {
                "text": "no, i just need to make sure it's cheap. oh, and i need parking",
                "metadata": {}
            },
            {
                "text": "I found 1 cheap hotel for you that includes parking. Do you like me to book it?",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [],
                            "stay": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "yes",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                }
            },
            {
                "text": "Yes, please. 6 people 3 nights starting on tuesday.",
                "metadata": {}
            },
            {
                "text": "I am sorry but I wasn't able to book that for you for Tuesday. Is there another day you would like to stay or perhaps a shorter stay?",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [],
                            "stay": "3",
                            "day": "tuesday",
                            "people": "6"
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "yes",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                }
            },
            {
                "text": "how about only 2 nights.",
                "metadata": {}
            },
            {
                "text": "Booking was successful.\nReference number is : 7GAWK763. Anything else I can do for you?",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [
                                {
                                    "name": "the cambridge belfry",
                                    "reference": "7GAWK763"
                                }
                            ],
                            "stay": "2",
                            "day": "tuesday",
                            "people": "6"
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "yes",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                }
            },
            {
                "text": "No, that will be all. Good bye.",
                "metadata": {}
            },
            {
                "text": "Thank you for using our services.",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [
                                {
                                    "name": "the cambridge belfry",
                                    "reference": "7GAWK763"
                                }
                            ],
                            "stay": "2",
                            "day": "tuesday",
                            "people": "6"
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "yes",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                }
            }
        ]
    }
    dialogue2 = {
        "log": [
            {
                "text": "am looking for a place to to stay that has cheap price range it should be in a type of hotel"
            },
            {
                "text": "Okay, do you have a specific area you want to stay in?",
                "metadata": default_state()['belief_state']
            },
            {
                "text": "no, i just need to make sure it's cheap. oh, and i need parking"
            },
            {
                "text": "I found 1 cheap hotel for you that includes parking. Do you like me to book it?",
                "metadata": default_state()['belief_state']
            },
            {
                "text": "Yes, please. 6 people 3 nights starting on tuesday."
            },
            {
                "text": "I am sorry but I wasn't able to book that for you for Tuesday. Is there another day you would like to stay or perhaps a shorter stay?",
                "metadata": default_state()['belief_state']
            },
            {
                "text": "how about only 2 nights."
            },
            {
                "text": "Booking was successful.\nReference number is : 7GAWK763. Anything else I can do for you?",
                "metadata": default_state()['belief_state']
            },
            {
                "text": "No, that will be all. Good bye."
            },
            {
                "text": "Thank you for using our services.",
                "metadata": default_state()['belief_state']
            }
        ]
    }
    create_data(dialogue2)
    print('Create WOZ-like dialogues. Get yourself a coffee, this might take a while.')
    delex_data = createData()
    print('Divide dialogues...')
    divideData(delex_data)


if __name__ == "__main__":
    main()
