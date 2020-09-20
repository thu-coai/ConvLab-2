import re
import torch


def is_slot_da(da):
    tag_da = {'Inform', 'Select', 'Recommend', 'NoOffer', 'NoBook', 'OfferBook', 'OfferBooked', 'Book'}
    not_tag_slot = {'Internet', 'Parking', 'none'}
    if da[0].split('-')[1] in tag_da and da[1] not in not_tag_slot:
        return True
    return False


def calculateF1(predict_golden,split=False):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        if split:
            #print('overall predicts = ',predicts)
            predicts = [[x[0], x[1], x[2].lower()] for x in predicts]
        else:
            predicts = [x.lower() for x in predicts]

        #print('predicts = ',predicts)
        labels = item['golden']

        if split:
            #print('overall label = ',labels)
            #print('='*100)
            labels = [[x[0], x[1], x[2].lower()] for x in labels]
        else:
            labels = [x.lower() for x in labels]
        
        #print('labels = ',labels)
        #print('='*100)
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


def tag2triples(word_seq, tag_seq):
    assert len(word_seq)==len(tag_seq)
    triples = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            domain, slot = tag[2:].split('+')
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    value += ' ' + word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            triples.append([domain, slot, value])
        i += 1
    return triples


def intent2triples(intent_seq):
    triples = []
    for intent in intent_seq:
        intent, slot, value = re.split('[+*]', intent)
        triples.append([intent, slot, value])
    return triples


def recover_intent(dataloader, intent_logits,req_logits, tag_logits, tag_mask_tensor, ori_word_seq, new2ori):
    max_seq_len = tag_logits.size(1)
    intents = []
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0:
            intents.append(dataloader.id2intent[j])

    reqs = []
    for j in range(dataloader.req_dim):
        if req_logits[j]>0:
            reqs.append(dataloader.id2req[j])

    tag_intent = []
    
    for i in range(dataloader.slot_intent_dim):
        tags = []
        intent = dataloader.id2slotintent[i]
        domain,base_intent = intent.split('-')
        if domain == 'general' or base_intent == 'Request':
            print('bug! get wrong intent')
            continue

        for j in range(3, max_seq_len-1):
            if tag_mask_tensor[i][j] == 1:
                value, tag_id = torch.max(tag_logits[i][j], dim=-1)
                tag = dataloader.id2tag[tag_id.item()]
                if tag != 'O':
                    prefix,slot = tag.split('-')
                    real_tag = prefix + '-' + intent + '+' + slot
                    tags.append(real_tag)
                else:
                    tags.append('O')

        if tags != []:
            recover_tags = []
            for i, tag in enumerate(tags):
                if new2ori[i] >= len(recover_tags):
                    recover_tags.append(tag)
            tag_intent += tag2triples(ori_word_seq, recover_tags)


    overall = []
    for i in intents:
        if i.split('-')[0] == 'general':
            overall.append([i,'none','none'])

    for i in reqs:
        domain,slot = i.split('-')
        intent = domain+'-'+'Request'
        if intent in intents:
            overall.append([intent,slot,'?'])

    for i in tag_intent:
        intent=i[0]
        slot = i[1]
        if intent in intents: 
            overall.append([intent, i[1],i[2]])

    return intents, reqs, tags,overall
