import re
import torch


def is_slot_da(da):
    tag_da = {'Inform', 'Recommend'}
    not_tag_slot = '酒店设施'
    if da[0] in tag_da and not_tag_slot not in da[2]:
        return True
    return False


def calculateF1(predict_golden,split=False):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        if split:
            #print('ori predicts = ',predicts)
            #print('ori labels = ',labels)
            predicts = [[x[0],x[1],x[2]] for x in predicts]
            labels = [[x[0],x[1],x[2]] for x in labels]
            #print('overall predicts = ',predicts)
            #print('overall labels = ',labels)
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                FP += 1
        for ele in labels:
            if ele not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1


def tag2das(word_seq, tag_seq):
    assert len(word_seq)==len(tag_seq)
    das = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            intent,slot = tag[2:].split('+')
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    # tag_seq[j][2:].split('+')[-1]==slot or tag_seq[j][2:] == tag[2:]
                    if word_seq[j].startswith('##'):
                        value += word_seq[j][2:]
                    else:
                        value += word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            das.append([intent,slot, value])
        i += 1
    return das


def intent2das(intent_seq):
    triples = []
    for intent in intent_seq:
        intent, domain, slot, value = re.split('\+', intent)
        triples.append([intent, domain, slot, value])
    return triples


def recover_intent(dataloader, intent_logits,req_logits,tag_logits, tag_mask_tensor, ori_word_seq, new2ori):
    # tag_logits = [sequence_length, tag_dim]
    # intent_logits = [intent_dim]
    # tag_mask_tensor = [sequence_length]
    #max_seq_len = tag_logits.size(0)
    max_seq_len = tag_logits.size(1)
    intents = []
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0:
            intents.append(dataloader.id2intent[j])
    
    reqs = []
    for j in range(dataloader.req_dim):
        if req_logits[j] > 0:
            reqs.append(dataloader.id2req[j])

    tag_intent = []
    for i in range(dataloader.slot_intent_dim):
        tags = []
        intent = dataloader.id2slotintent[i]
        domain,base_intent = intent.split('-')
        for j in range(3,max_seq_len-1):
            if tag_mask_tensor[i][j] == 1:
                value,tag_id = torch.max(tag_logits[i][j],dim=-1)
                tag = dataloader.id2tag[tag_id.item()]
                if tag != 'O':
                    prefix,slot = tag.split('-')
                    real_tag = prefix + '-' + intent + '+' + slot
                    tags.append(real_tag)
                else:
                    tags.append('O')
        if tags != []:
            tag_intent += tag2das(ori_word_seq,tags) 

    overall = []
    for i in intents:
        if i.split('-')[-1] == 'General':
            overall.append([i,'none','none'])   

    for i in reqs:
        domain,slot = i.split('-')
        intent = domain + '-' + 'Request'
        if intent in intents:
            overall.append([intent,slot,''])

    for i in tag_intent:
        intent = i[0]
        slot = i[1]
        if intent in intents:
            overall.append([intent,i[1],i[2]])
    '''
    das = []
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0:
            intent, domain, slot, value = re.split('\+', dataloader.id2intent[j])
            das.append([intent, domain, slot, value])
    tags = []
    for j in range(1 , max_seq_len -1):
        if tag_mask_tensor[j] == 1:
            value, tag_id = torch.max(tag_logits[j], dim=-1)
            tags.append(dataloader.id2tag[tag_id.item()])
    tag_intent = tag2das(ori_word_seq, tags)
    das += tag_intent
    return das
    '''
    return intents,reqs,tags,overall
