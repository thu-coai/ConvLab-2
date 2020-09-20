import json
import os
import zipfile
import sys
from collections import Counter
from transformers import BertTokenizer


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def preprocess(mode):
    assert mode == 'all' or mode == 'usr' or mode == 'sys'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../data/crosswoz')
    processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_intent = []
    all_tag = []
    all_used_tag = []
    all_req = []

    context_size = 3

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    cnt = 0
    for key in data_key:    
        processed_data[key] = []
        for no, sess in data[key].items():
            context = []
            for i, turn in enumerate(sess['messages']):
                if mode == 'usr' and turn['role'] == 'sys':
                    context.append(turn['content'])
                    continue
                elif mode == 'sys' and turn['role'] == 'usr':
                    context.append(turn['content'])
                    continue
                utterance = turn['content']
                # Notice: ## prefix, space remove
                tokens = tokenizer.tokenize(utterance)
                golden = []#overall label

                span_info = []
                intents = []
                reqs = []
                for intent, domain, slot, value in turn['dialog_act']:#FIXME:span抽取逻辑
                    slot = slot.replace('-','_')
                    if intent.capitalize() == 'Request':
                        reqs.append(domain+'-'+slot)
                    intents.append('-'.join([domain,intent]))
                    if intent in ['Inform', 'Recommend'] and '酒店设施' not in slot:
                        if value in utterance:
                            idx = utterance.index(value)
                            idx = len(tokenizer.tokenize(utterance[:idx]))
                            #'domain-intent+slot'
                            span_info.append(('-'.join([domain,intent])+'+'+slot,idx,idx+len(tokenizer.tokenize(value)), value))
                            token_v = ''.join(tokens[idx:idx+len(tokenizer.tokenize(value))])
                            token_v = token_v.replace('##', '')
                            golden.append([domain+'-'+intent, slot, token_v])
                        else:
                            cnt += 1
                            golden.append([domain+'-'+intent, slot, value])
                    else:
                        #intents.append('+'.join([intent, domain, slot, value]))#为什么intent in ['Inform','Recommend']的只提取了酒店设施
                        golden.append([domain+'-'+intent, slot, value])

                tags = []
                used_tags = []
                for j, _ in enumerate(tokens):
                    for span in span_info:
                        if j == span[1]:
                            tag = "B-" + span[0]
                            tags.append(tag)
                            used_tags.append("B-"+tag.split('+')[-1])#去掉domain-intent, 只保留slot
                            break
                        if span[1] < j < span[2]:
                            tag = "I-" + span[0]
                            tags.append(tag)
                            used_tags.append("I-"+tag.split('+')[-1])
                            break
                    else:
                        tags.append("O")
                        used_tags.append("O")
                processed_data[key].append([tokens, tags, intents, golden, context[-context_size:],reqs])

                all_intent += intents
                all_tag += tags
                all_used_tag += used_tags
                all_req += reqs

                context.append(turn['content'])

        all_intent = [x[0] for x in dict(Counter(all_intent)).items()]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items()]
        all_req = [x[0] for x in dict(Counter(all_req)).items()]
        all_used_tag = [x[0] for x in dict(Counter(all_used_tag)).items()]
        print('loaded {}, size {}'.format(key, len(processed_data[key])))
        json.dump(processed_data[key], open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False)
    
    all_used_intent = []
    for i in all_intent:
        domain,intent = i.split('-')
        if intent not in  ['General','Request']:
            all_used_intent.append(i)

    print('cnt = ',cnt)
    print('sentence label num:', len(all_intent))
    print('tag num:', len(all_tag))
    print(all_intent)
    json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(all_req,open(os.path.join(processed_data_dir,'req_vocab.json'), 'w', encoding='utf-8'),indent=2,ensure_ascii=False)
    json.dump(all_tag, open(os.path.join(processed_data_dir, 'ori_tag_vocab.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(all_used_tag,open(os.path.join(processed_data_dir,'tag_vocab.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(all_used_intent,open(os.path.join(processed_data_dir,'slot_intent_vocab.json'),'w',encoding='utf-8'),indent=2,ensure_ascii=False)

if __name__ == '__main__':  
    mode = sys.argv[1]
    preprocess(mode)
