import numpy as np
import torch
import random
from transformers import BertTokenizer
import math
from collections import Counter


class Dataloader:
    def __init__(self, intent_vocab, tag_vocab,req_vocab,slot_intent_vocab,pretrained_weights):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param req_vocab: list of all slots of intent request
        :param pretrained_weights: which bert, e.g. 'bert-base-uncased'
        """
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.req_vocab = req_vocab
        self.slot_intent_vocab = slot_intent_vocab

        self.intent_dim = len(intent_vocab)
        self.slot_intent_dim = len(slot_intent_vocab)
        self.tag_dim = len(tag_vocab)
        self.req_dim =  len(req_vocab)

        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2slotintent = dict([(i,x) for i, x in enumerate(slot_intent_vocab)])
        self.slotintent2id = dict([(x,i) for i,x in enumerate(slot_intent_vocab)])
        self.id2req = dict([(i, x) for i, x in enumerate(req_vocab)])
        self.req2id = dict([(x, i) for i, x in enumerate(req_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)
        self.req_weight = [1]*len(self.req2id)

    def tag_transfer(self,tags):

        result = [['O']*len(tags) for i in range(self.slot_intent_dim)]
        for i,tag in enumerate(tags):
            if tag != 'O':
                intent,value = tag.split('+')
                prefix = intent[0]
                intent = intent[2:]
                if intent in self.slotintent2id.keys():
                    intent_id = self.slotintent2id[intent]
                    result[intent_id][i] = prefix + '-' + value

        return result

    def load_data(self, data, data_key, cut_sen_len, use_bert_tokenizer=True):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/test
        :param data:
        :return:
        """
        self.data[data_key] = data
        max_sen_len, max_context_len = 0, 0
        sen_len = []
        context_len = []
        for d in self.data[data_key]:
            max_sen_len = max(max_sen_len, len(d[0]))
            sen_len.append(len(d[0]))
    
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"], context(list of str))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]#text
                d[1] = d[1][:cut_sen_len]#bio tag 

                #d[2] intent
                #d[3] overall label(dialog act)
                d[4] = [' '.join(s.split()[:cut_sen_len]) for s in d[4]] #d[4] context
                d[5] = d[5][:cut_sen_len] #slot of req
            d[4] = self.tokenizer.encode('[CLS] ' + ' [SEP] '.join(d[4]))
            max_context_len = max(max_context_len, len(d[4]))
            context_len.append(len(d[4]))

            if use_bert_tokenizer:
                word_seq, tag_seq, new2ori = self.bert_tokenize(d[0], d[1])
            else:
                word_seq = d[0]
                tag_seq = self.tag_transfer(d[1])
                new2ori = None
            d.append(self.seq_req2id(d[5]))
            d.append(new2ori)
            d.append(word_seq)
            tag_id_seq = []
            for i in tag_seq:
                tag_id_seq.append(self.seq_tag2id(i))
            d.append(tag_id_seq)
            d.append(self.seq_intent2id(d[2]))
            # ori d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), new2ori, new_word_seq, tag2id_seq, intent2id_seq)
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), reqs,req2id_seq, new2ori, new_word_seq, tag2id_seq, intent2id_seq)

            if data_key=='train':
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
                for req_id in d[-5]:
                    self.req_weight[req_id] += 1

        if data_key == 'train':
            train_size = len(self.data['train'])
            for intent, intent_id in self.intent2id.items():
                neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
            self.intent_weight = torch.tensor(self.intent_weight)

            for req,req_id in self.req2id.items():
                neg_pos = (train_size - self.req_weight[req_id]) / self.req_weight[req_id]
                self.req_weight[req_id] = np.log10(neg_pos)
            self.req_weight = torch.tensor(self.req_weight)

        print('max sen bert len', max_sen_len)
        print(sorted(Counter(sen_len).items()))
        print('max context bert len', max_context_len)
        print(sorted(Counter(context_len).items()))

    def bert_tokenize(self, word_seq, tag_seq):
        split_tokens = []
        new_tag_seq = []  
        new2ori = {}
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(' '.join(word_seq))
        accum = ''
        i, j = 0, 0

        all_tag = [[] for i in range(self.slot_intent_dim)]

        for i, token in enumerate(basic_tokens): 
            if (accum + token).lower() == word_seq[j].lower():
                accum = ''
            else:
                accum += token
            for sub_token in self.tokenizer.wordpiece_tokenizer.tokenize(basic_tokens[i]):
                new2ori[len(new_tag_seq)] = j
                split_tokens.append(sub_token)
                
                tag = tag_seq[j]  
                for k in range(self.slot_intent_dim):
                    all_tag[k].append('O')
                if tag != 'O':
                    intent,value = tag.split('+')
                    prefix = intent[0]
                    intent = intent[2:]
                    if intent in self.slotintent2id.keys():
                        intent_id = self.slotintent2id[intent]
                        all_tag[intent_id][-1] = prefix + '-' + value     
                new_tag_seq.append(tag_seq[j])

            if accum == '':
                j += 1
        return split_tokens, all_tag, new2ori

    def seq_tag2id(self, tags):
        return [self.tag2id[x] if x in self.tag2id else self.tag2id['O'] for x in tags]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]
    
    def seq_slotintent2id(self,intents):
        return [self.slotintent2id[x] for x in intents if x in self.slotintent2id]
    
    def seq_id2slotintent(self,ids):
        return [self.id2slotintent[x] for x in ids]

    def seq_id2req(self,ids):
        return [self.id2req[x] for x in ids]

    def seq_req2id(self, reqs):
        return [self.req2id[x] for x in reqs if x in self.req2id]

    def pad_batch(self, batch_data):
        #print('enter pad batch')
        batch_size = len(batch_data)
        #print('batch_size = ',batch_size)
        max_seq_len = max([len(x[-3]) for x in batch_data]) + 2 
        #print('max_seq_len = ',max_seq_len)
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        
        tag_mask_tensor = torch.zeros((batch_size*self.slot_intent_dim, max_seq_len+2), dtype=torch.long) 
        base_tag_mask_tensor = torch.zeros((batch_size*self.slot_intent_dim, max_seq_len+2),dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size*self.slot_intent_dim, max_seq_len+2), dtype=torch.long)

        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        req_tensor = torch.zeros((batch_size, self.req_dim), dtype=torch.float)

        context_max_seq_len = max([len(x[4]) for x in batch_data])#max([len(x[-5]) for x in batch_data])
        context_mask_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_seq_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        for i in range(batch_size):
            words = batch_data[i][-3]
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            reqs = batch_data[i][-5]
            words = ['[CLS]'] + words + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(words)
            sen_len = len(words)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])

            flag = False
            for j in range(self.slot_intent_dim):
                tag_seq_tensor[i*self.slot_intent_dim+j, 3:sen_len+1] = torch.LongTensor(tags[j])
                base_tag_mask_tensor[i*self.slot_intent_dim+j,3:sen_len+1] = torch.LongTensor([1] * (sen_len-2))
                if tags[j] != [self.tag2id['O']]*len(tags[j]): 
                    tag_mask_tensor[i*self.slot_intent_dim+j, 3:sen_len+1] = torch.LongTensor([1] * (sen_len-2))

            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            for j in intents:
                intent_tensor[i, j] = 1.            
            for j in reqs:
                req_tensor[i,j] = 1.

            context_len = len(batch_data[i][4])#len(batch_data[i][-5])
            context_seq_tensor[i, :context_len] = torch.LongTensor([batch_data[i][4]])#[batch_data[i][-5]])
            context_mask_tensor[i, :context_len] = torch.LongTensor([1] * context_len)

        return word_seq_tensor, tag_seq_tensor, intent_tensor, req_tensor, word_mask_tensor, tag_mask_tensor, base_tag_mask_tensor,context_seq_tensor, context_mask_tensor

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size:(i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
