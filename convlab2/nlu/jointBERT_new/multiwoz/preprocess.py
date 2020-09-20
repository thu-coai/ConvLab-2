import json
import os
import zipfile
import sys
from collections import Counter


def read_zipped_json(filepath, filename):
    print("zip file path = ", filepath)
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def da2triples(dialog_act):
    triples = []
    for intent, svs in dialog_act.items():
        for slot, value in svs:
            triples.append([intent, slot, value])
    return triples


def preprocess(mode):
    assert mode == 'all' or mode == 'usr' or mode == 'sys'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../data/multiwoz')
    processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_da = []
    all_intent = []
    all_req = []
    all_tag = []
    all_used_tag = []#去掉domain,intent
    context_size = 3

    other = {}

    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            context = []
            for is_sys, turn in enumerate(sess['log']):
                if mode == 'usr' and is_sys % 2 == 1:
                    context.append(turn['text'])
                    continue
                elif mode == 'sys' and is_sys % 2 == 0:
                    context.append(turn['text'])
                    continue
                tokens = turn["text"].split()
                dialog_act = {}
                for dacts in turn["span_info"]:
                    if dacts[0] not in dialog_act:
                        dialog_act[dacts[0]] = []
                    dialog_act[dacts[0]].append([dacts[1], " ".join(tokens[dacts[3]: dacts[4] + 1])])


                spans = turn["span_info"]
                tags = []
                used_tags = []
                for i, _ in enumerate(tokens):
                    for span in spans:
                        if i == span[3]:
                            domain,intent = span[0].split('-')
                            tags.append("B-" + domain + '-' +intent.capitalize() + "+" + span[1].capitalize()) #修改: B/I-domain-intent-slot
                            used_tags.append("B-" + span[1].capitalize())
                            break
                        if span[3] < i <= span[4]:
                            domain,intent = span[0].split('-')
                            tags.append("I-" + domain + '-' +intent.capitalize() + "+" + span[1].capitalize()) #修改: B/I-domain-intent-slot
                            used_tags.append("I-" + span[1].capitalize())
                            break
                    else:
                        tags.append("O")
                        used_tags.append('O')

                reqs = []
                intents = []
                intent_cnt = {}
                for dacts in turn["dialog_act"]: #domain-intent
                    processed_da = []
                    for i, dact in enumerate(turn["dialog_act"][dacts]): #slot, slot_value
                        temp_domain, temp_intent = dacts.split('-')

                        if temp_domain == 'general' and dacts not in intents:
                            intents.append(dacts)
                            processed_da.append(dact)
                        elif temp_intent == 'Request' and dacts not in intents:
                            intents.append(dacts)
                            reqs.append(dacts.split('-')[0]+'-'+dact[0]) 
                            processed_da.append(dact)
                                                        
                        elif temp_intent == 'Inform':
                            slot = dact[0] 

                            temp_key = dacts + '-' + slot
                            if temp_key not in intent_cnt.keys(): 
                                intent_cnt[temp_key] = 0
                            
                            #到span_info中check 并做替换
                            if dacts in dialog_act:#有相同的domain-intent
                                possible_value = [s[1] for s in dialog_act[dacts] if s[0].lower() == slot.lower()] #相同的slot
                                if len(possible_value): #有相同的slot
                                    if dacts not in intents: #加到intents中
                                        assert(intent_cnt[temp_key] == 0)
                                        intents.append(dacts)

                                    #替换dialog act
                                    processed_da.append([slot, possible_value[intent_cnt[temp_key] if intent_cnt[temp_key] < len(possible_value) else -1]]) #按顺序替换　如果不足　就选最后一个)
                                    intent_cnt[temp_key] += 1 

                    turn['dialog_act'][dacts] = processed_da

                processed_data[key].append([tokens, tags, intents, da2triples(turn["dialog_act"]), context[-context_size:],reqs])
                all_da += [da for da in turn['dialog_act']]
                all_intent += intents
                all_req += reqs
                all_tag += tags
                all_used_tag += used_tags
                

                context.append(turn['text'])

        all_da = [x[0] for x in dict(Counter(all_da)).items() if x[1]]
        all_intent = [x[0] for x in dict(Counter(all_intent)).items() if x[1]]
        all_req = [x[0] for x in dict(Counter(all_req)).items() if x[1]]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items() if x[1]]
        all_used_tag = [x[0] for x in dict(Counter(all_used_tag)).items() if x[1]]

        print('loaded {}, size {}'.format(key, len(processed_data[key])))
        json.dump(processed_data[key], open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w'), indent=2)


    print('dialog act num:', len(all_da))
    print('sentence label num:', len(all_intent))
    print('req slot num:', len(all_req))
    print('tag num:', len(all_tag))
    print('used tag num:',len(all_used_tag))

    all_used_intent = []
    for i in all_intent:
        domain,intent = i.split('-')

        if domain != 'general' and intent != 'Request':
            all_used_intent.append(i)

    
    json.dump(all_da, open(os.path.join(processed_data_dir, 'all_act.json'), 'w'), indent=2)
    json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w'), indent=2)
    json.dump(all_req, open(os.path.join(processed_data_dir, 'req_vocab.json'), 'w'), indent=2)
    json.dump(all_tag, open(os.path.join(processed_data_dir, 'ori_tag_vocab.json'), 'w'), indent=2)
    json.dump(all_used_tag,open(os.path.join(processed_data_dir,'tag_vocab.json'),'w'), indent=2)
    json.dump(all_used_intent,open(os.path.join(processed_data_dir,'slot_intent_vocab.json'),'w'), indent=2)

if __name__ == '__main__':
    mode = sys.argv[1]
    preprocess(mode)
