import os
import json
import pickle
import torch
from pytorch_pretrained_bert import BertTokenizer
from collections import defaultdict
### this file is to convert the raw woz data into the format required for prepross.py
bert=True
bert_type='bert-large-uncased'
tokenizer=BertTokenizer.from_pretrained(bert_type)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            

cldict={'pricerange':'price range','leaveat':'leave at','arriveby':'arrive by'}
def preslot(strl):
    s=strl.split('-')
    k=s[1].split(' ')
    if len(k)==1 and k[0] in cldict:
        return s[0]+' - '+cldict[k[0]]
    if len(k)==1:
        return s[0]+' - '+k[0]
    else:
        return s[0]+' - '+k[0]+' '+k[1]


def convert_data():
    path = 'data/'
    sl_dict = json.load(open(os.path.join(path, 'mwoz2_sl.dict')))
    dm_dict = json.load(open(os.path.join(path, 'mwoz2_dm.dict')))
    mode = 'real'
    fp = os.path.join(path, '{}_dials.json'.format(mode))
    with open(fp, 'r') as json_file:
        data = json.load(json_file)

    data_input = []
    for s in data:
        sdict = {'user_input': [],
                 'system_input': [],
                 'belief_input': [],
                 'labeld': [],
                 'labels': [],
                 'labelv': [],
                 }

        prev_label = []
        prev_labelv = [[]]
        texta = []
        belief_input = ['[CLS]', '[SEP]']
        for i, turn in enumerate(s['dialogue']):
            svs = defaultdict(list)
            for st in turn['belief_state']:  # loop slot-value pairs
                if st['act'] == 'inform':
                    dsl, val = st['slots'][0][0], st['slots'][0][1]
                    if val == 'dontcare':
                        val = "do not care"
                    dm = dsl.split('-')[0]
                    sl = dsl.split('-')[1]

                    if sl in list(cldict.keys()):
                        sl = cldict[sl]
                    svs[dm].append((sl, val))
                    svs[dm].sort(key=lambda x: sl_dict[dm][x[0]], reverse=True)
            svs = sorted(svs.items(), key=lambda x: dm_dict[x[0]], reverse=True)

            sdict['belief_input'].append(belief_input)

            labeld = ['[CLS]']
            labels = [['[CLS]', '[SEP]']]
            labelv = [[['[CLS]', '[SEP]'], ['[CLS]', '[SEP]']]]
            belief_input = ['[CLS]']
            for dsv in svs:
                labeld.append(dsv[0])
                labels.append(['[CLS]'])
                labelv.append([['[CLS]', '[SEP]']])
                belief_input.extend([dsv[0]] + ['-'])
                for sv in dsv[1]:
                    labels[-1].append(sv[0])
                    labelv[-1].append(['[CLS]'] + tokenizer.tokenize(sv[1]) + ['[SEP]'])
                    belief_input.extend([sv[0]] + [','] + tokenizer.tokenize(sv[1]) + [';'])
                labels[-1].append('[SEP]')
                labelv[-1].append(['[CLS]', '[SEP]'])
            labeld.append('[SEP]')
            labels.append(['[CLS]', '[SEP]'])
            labelv.append([['[CLS]', '[SEP]'], ['[CLS]', '[SEP]']])

            belief_input.append('[SEP]')

            assert len(labels) == len(labelv)
            sdict['labeld'].append(labeld)
            sdict['labels'].append(labels)
            sdict['labelv'].append(labelv)
            a = turn['system_transcript']
            b = turn['transcript']
            if bert:
                ta = ['[CLS]', 'system', ':'] + tokenizer.tokenize(a) + ['[SEP]']
                tb = ['[CLS]', 'user', ':'] + tokenizer.tokenize(b) + ['[SEP]']
            sdict['system_input'].append(ta)
            sdict['user_input'].append(tb)

        data_input.append(sdict)

    print(len(data_input))
    print(data_input[0]['belief_input'])

    fp = os.path.join(path, 'mwoz2_format_{}.json'.format(mode))
    with open(fp, 'w') as f:
        for d in data_input:
            f.write(json.dumps(d) + '\n')


switch=0
if __name__=='__main__':
    convert_data()
    #
    # path ='data/'
    # fp = os.path.join(path, '{}_dials.json'.format('train'))
    # with open(fp, 'r') as json_file:
    #     data = json.load(json_file)
    #
    # sl_dict=defaultdict(dict)
    # dm_dict={}
    # dsl_dict={}
    # num_turn=0
    # num_sv=0
    # for s in data:
    #     for i,turn in enumerate(s['dialogue']):
    #         num_turn+=1
    #         for st in turn['belief_state']: # loop slot-value pairs
    #             if st['act'] == 'inform':
    #                 dsl, val = st['slots'][0][0], st['slots'][0][1]
    #                 num_sv+=1
    #                 if val == 'dontcare':
    #                     val = "do not care"
    #                 dm=dsl.split('-')[0]
    #                 sl=dsl.split('-')[1]
    #                 if sl in list(cldict.keys()):
    #                     sl=cldict[sl]
    #                 dm_dict[dm] = dm_dict.get(dm, 0) + 1
    #                 sl_dict[dm][sl] = sl_dict[dm].get(sl, 0) + 1
    #                 dsl_dict[dsl] = dsl_dict.get(dsl, 0) + 1
    # print(sl_dict)
    # print(len(sl_dict))
    # print(dm_dict)
    # print(len(dm_dict))
    #
    # print(dsl_dict)
    # print(len(dsl_dict))
    #
    # print('avg sv/turn:', num_sv/num_turn)
    #
    #
    # patho =path
    # fp = os.path.join(patho, 'mwoz2_sl.dict')
    # with open(fp, 'w') as f:
    #     f.write(json.dumps(sl_dict) + '\n')
    # fp = os.path.join(patho, 'mwoz2_dm.dict')
    # with open(fp, 'w') as f:
    #     f.write(json.dumps(dm_dict) + '\n')
    #
    #
    # for mode in ['train', 'dev', 'test']:
    #     fp = os.path.join(path, '{}_dials.json'.format(mode))
    #     with open(fp, 'r') as json_file:
    #         data = json.load(json_file)
    #
    #
    #     data_input = []
    #     for s in data:
    #         sdict = {'user_input':[],
    #                  'system_input':[],
    #                  'belief_input':[],
    #                  'labeld':[],
    #                  'labels':[],
    #                  'labelv':[],
    #                 }
    #
    #         prev_label = []
    #         prev_labelv = [[]]
    #         texta=[]
    #         belief_input=['[CLS]','[SEP]']
    #         for i,turn in enumerate(s['dialogue']):
    #             svs = defaultdict(list)
    #             for st in turn['belief_state']: # loop slot-value pairs
    #                 if st['act'] == 'inform':
    #                     dsl, val = st['slots'][0][0], st['slots'][0][1]
    #                     if val == 'dontcare':
    #                         val = "do not care"
    #                     dm=dsl.split('-')[0]
    #                     sl=dsl.split('-')[1]
    #
    #                     if sl in list(cldict.keys()):
    #                         sl=cldict[sl]
    #                     svs[dm].append((sl, val))
    #                     svs[dm].sort(key=lambda x: sl_dict[dm][x[0]],reverse=True)
    #             svs=sorted(svs.items(),key=lambda x: dm_dict[x[0]],reverse=True)
    #
    #
    #
    #             sdict['belief_input'].append(belief_input)
    #
    #             labeld = ['[CLS]']
    #             labels=[['[CLS]','[SEP]']]
    #             labelv=[[['[CLS]','[SEP]'],['[CLS]','[SEP]']]]
    #             belief_input=['[CLS]']
    #             for dsv in svs:
    #                 labeld.append(dsv[0])
    #                 labels.append(['[CLS]'])
    #                 labelv.append([['[CLS]','[SEP]']])
    #                 belief_input.extend([dsv[0]]+['-'])
    #                 for sv in dsv[1]:
    #                     labels[-1].append(sv[0])
    #                     labelv[-1].append(['[CLS]']+tokenizer.tokenize(sv[1])+['[SEP]'])
    #                     belief_input.extend([sv[0]]+[',']+tokenizer.tokenize(sv[1])+[';'])
    #                 labels[-1].append('[SEP]')
    #                 labelv[-1].append(['[CLS]','[SEP]'])
    #             labeld.append('[SEP]')
    #             labels.append(['[CLS]','[SEP]'])
    #             labelv.append([['[CLS]','[SEP]'],['[CLS]','[SEP]']])
    #
    #             belief_input.append('[SEP]')
    #
    #             assert len(labels)==len(labelv)
    #             sdict['labeld'].append(labeld)
    #             sdict['labels'].append(labels)
    #             sdict['labelv'].append(labelv)
    #             a=turn['system_transcript']
    #             b=turn['transcript']
    #             if bert:
    #                 ta =['[CLS]','system',':']+tokenizer.tokenize(a)+['[SEP]']
    #                 tb = ['[CLS]','user',':']+tokenizer.tokenize(b)+['[SEP]']
    #             sdict['system_input'].append(ta)
    #             sdict['user_input'].append(tb)
    #
    #         data_input.append(sdict)
    #
    #
    #     print(len(data_input))
    #     print(data_input[0]['belief_input'])
    #
    #     fp = os.path.join(patho, 'mwoz2_format_{}.json'.format(mode))
    #     with open(fp, 'w') as f:
    #         for d in data_input:
    #             f.write(json.dumps(d) + '\n')
