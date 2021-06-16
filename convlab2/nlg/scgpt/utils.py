# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:34:55 2020

@author: truthless
"""

def tuple2dict(t):
    '''
    tuple: [(intent, domain, slot, value)]
    dict: [domain: { intent: [slot, value] }]
    '''
    d = {}
    for intent, domain, slot, value in t:
        if domain not in d:
            d[domain] = {}
        if intent not in d[domain]:
            d[domain][intent] = []
        if slot == 'none' or slot is None:
            continue
        d[domain][intent].append([slot, value])
    return d

def dict2dict(D):
    '''
    dict: [domain-intent: [slot, value]]
    dict: [domain: { intent: [slot, value] }]
    '''
    d = {}
    for domint in D:
        domain, intent = domint.split('-')
        if domain not in d:
            d[domain] = {}
        if intent not in d[domain]:
            d[domain][intent] = []
        for slot, value in D[domint]:
            if slot == 'none' or slot is None:
                continue
            d[domain][intent].append([slot, value])
    return d

def dict2seq(d):
    '''
    dict: [domain: { intent: [slot, value] }]
    seq: [domain { intent ( slot = value ; ) @ } | ]
    '''
    s = ''
    first_domain = True
    first_intent = True
    first_slot = True
    for domain in d:
        if not first_domain:
            s += ' | '
        s += domain
        s += ' { '
        first_domain = False
        first_intent = True
        for intent in d[domain]:
            if not first_intent:
                s += ' @ '
            s += intent
            s += ' ( '
            first_intent = False
            first_slot = True
            for slot, value in d[domain][intent]:
                if not first_slot:
                    s += ' ; '
                s += slot
                if value:
                    s += ' = '
                    s += value
                first_slot = False
            s += ' )'
        s += ' }'
    return s.lower()

def tuple2seq(t):
    d = tuple2dict(t)
    s = dict2seq(d)
    return s
    
if __name__ == '__main__':
    da_tuple = [('Inform', 'Booking', 'none', 'none'), ('Inform', 'Hotel', 'Price', 'cheap'), ('Inform', 'Hotel', 'Choice', '1'), ('Inform', 'Hotel', 'Parking', 'none')]
    da_dict = tuple2dict(da_tuple)
    print(da_dict)
    da_seq = dict2seq(da_dict)
    print(da_seq)

    da_tuple = [('Request', 'Hotel', 'Address', '?'), ('Request', 'Hotel', 'Area', '?'), ('Inform', 'Attraction', 'Area', 'center'), ('Inform', 'Hotel', 'Price', 'cheap')]
    da_dict = tuple2dict(da_tuple)
    print(da_dict)
    da_seq = dict2seq(da_dict)
    print(da_seq)
    
    D = {'Hotel-Inform': [['Price', 'cheap'], ['Type', 'hotel']]}
    da_dict = dict2dict(D)
    print(da_dict)
    
