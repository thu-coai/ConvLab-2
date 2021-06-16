# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:38:53 2020

@author: truthless
"""

import os
import json
from convlab2.nlg.scgpt.utils import dict2dict, dict2seq

cur_dir = os.path.dirname(os.path.abspath(__file__)) 
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        cur_dir)))), 'data/multiwoz/')

with open(os.path.join(data_dir, '0807_final.json'),'r', encoding='utf8') as f:
    data = json.load(f)

with open(os.path.join(data_dir, 'valListFile'), 'r') as f:
    val_list = f.read().splitlines()
with open(os.path.join(data_dir, 'testListFile'), 'r') as f:
    test_list = f.read().splitlines()
    
results = {}
results_val = {}
results_test = {}

for title, sess in data.items():
    logs = sess['log']
    turns = []
    turn = {'turn':0, 'sys':'', 'sys_da':''}
    current_domain = None
    for i, diag in enumerate(logs):
        text = diag['text']
        da = diag['dialog_act']
        span = diag['span_info']
        if i % 2 == 0:
            turn['usr'] = text
            if current_domain:
                da = eval(str(da).replace('Booking', current_domain))
                span = eval(str(span).replace('Booking', current_domain))
            turn['usr_da'] = da
            turn['usr_span'] = span
            turns.append(turn)
        else:
            turn = {'turn': i//2 +1}
            turn['sys'] = text
            turn['sys_da'] = da
            turn['sys_span'] = span
        for key in da:
            domain = key.split('-')[0]
            if domain not in ['general', 'Booking']:
                current_domain = domain
    title = title
    if title in val_list:
        current = results_val
    elif title in test_list:
        current = results_test
    else:
        current = results
    current[title] = turns
    
results = eval(str(results).replace(" n't", " not"))
results_val = eval(str(results_val).replace(" n't", " not"))
results_test = eval(str(results_test).replace(" n't", " not"))

def init_domain():
    return {'Attraction':False,
            'Hospital':False,
            'Hotel':False,
            'Police':False,
            'Restaurant':False,
            'Taxi':False,
            'Train':False}

def write_file(name, data):
    with open(f'{name}.txt', 'w', encoding='utf-8') as f:
        for ID in data:
            sess = data[ID]
            sess_domains = init_domain()
            for turn in sess:
                if not turn['usr_da']:
                    continue
                turn['usr_da'] = eval(str(turn['usr_da']).replace('Bus','Train'))
                da_seq = dict2seq(dict2dict(turn['usr_da'])).replace('&', 'and')
                domains = set([key.split('-')[0] for key in turn['usr_da'].keys()])
                for domain in domains:
                    if domain not in ['general', 'Booking'] and not sess_domains[domain]:
                        da_seq = da_seq.replace(domain.lower(), domain.lower()+' *', 1)
                        sess_domains[domain] = True
                da_uttr = turn['usr'].replace(' bus ', ' train ').replace('&', 'and')
                f.write(f'{da_seq} & {da_uttr}\n')

if not os.path.exists(os.path.join(cur_dir,'data')):
    os.makedirs(os.path.join(cur_dir, 'data'))
write_file(os.path.join(cur_dir, 'data/train'), dict(results, **results_val))
write_file(os.path.join(cur_dir, 'data/test'), results_test)
