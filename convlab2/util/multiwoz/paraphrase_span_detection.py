# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:49:53 2020

@author: truthless
"""

import spacy
from fuzzywuzzy import fuzz

digit2word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven',
    '12': 'twelve'
}
word2digit = {v:k for k,v in digit2word.items()}

nlp = spacy.load('en_core_web_sm')
threshold = 60

def digit_normalize(utt_list):
    for i, text in enumerate(utt_list):
        if text in word2digit:
            utt_list[i] = word2digit[text]
    return utt_list

def phrase_idx_utt(value_list, utt_list, constraint=[]):
    value_list = digit_normalize(value_list)
    utt_list = digit_normalize(utt_list)
    candidates = []
    l = len(value_list)
    for i in [l, l-1, l+1]:
        if i == 0:
            continue
        for j in range(len(utt_list)-i+1):
            if constraint and j <= constraint[0] and constraint[0] < j+i:
                if j == constraint[0]:
                    constraint.append(constraint.pop(0))
                continue
            score = fuzz.ratio(' '.join(utt_list[j:j+i]), ' '.join(value_list))
            if score > threshold:
                candidates.append((score, j, j+i-1))
    return sorted(candidates, key=lambda x:x[0], reverse=True)[0][1:] if candidates else None

def preprocess(utt, da):
    '''
    utt: str
    da: dict {'domain-intent': [slot, value]}
    '''
    with nlp.disable_pipes('tagger', 'parser'):
        '''
        get tokens, recover the paraphrased entity 
        '''
        tokens = [token.text for token in nlp(utt)]
        for key, pair in da.items():
            constraint = []
            intent = key.split('-')[1].lower()
            if intent not in ['inform']:
                continue
            for slot, value in pair:
                if slot.lower() in ['name', 'dest', 'depart']:
                    value_tokens = [token.text for token in nlp(value)]
                    span = phrase_idx_utt(value_tokens, tokens, constraint)
                    if span is not None:
                        for i in range(span[0], span[1]+1):
                            constraint.append(i)
                        tokens[span[0]:span[1]+1] = value_tokens
                        
        '''
        get labels, tag or slot or none
        '''
        labels = dict()
        for key, pair in da.items():
            constraint = []
            intent = key.split('-')[1].lower()
            if intent in ["request"]:
                labels[key] = []
            elif intent in ['inform']:
                labels[key] = ["O"] * len(tokens)
            else:
                labels[key] = None
            for slot, value in pair:
                if intent in ["request"]:
                    labels[key].append(slot)
                elif intent in ['inform']:
                    value_tokens = [token.text for token in nlp(value)]
                    span = phrase_idx_utt(value_tokens, tokens, constraint)
                    if span is not None:
                        for i in range(span[0], span[1]+1):
                            constraint.append(i)
                        #tags[span[0]] = "B-" + da[1] + '-' + da[0] + "+" + da[2]
                        labels[key][span[0]] = "B-" + slot
                        for i in range(span[0]+1, span[1]+1):
                            #tags[i] = "I-" + da[1] + '-' + da[0] + "+" + da[2]
                            labels[key][i] = "I-" + slot
    return tokens, labels
