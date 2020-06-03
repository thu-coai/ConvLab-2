# -*- coding: gbk -*-
"""
Evaluate NLU models on specified dataset
Usage: python evaluate.py [MultiWOZ|CrossWOZ] [TRADE|mdbt|sumbt|rule]
"""
import random
import numpy
import torch
import sys
from tqdm import tqdm
import copy
import jieba

multiwoz_slot_list = ['attraction-area', 'attraction-name', 'attraction-type', 'hotel-day', 'hotel-people', 'hotel-stay', 'hotel-area', 'hotel-internet', 'hotel-name', 'hotel-parking', 'hotel-pricerange', 'hotel-stars', 'hotel-type', 'restaurant-day', 'restaurant-people', 'restaurant-time', 'restaurant-area', 'restaurant-food', 'restaurant-name', 'restaurant-pricerange', 'taxi-arriveby', 'taxi-departure', 'taxi-destination', 'taxi-leaveat', 'train-people', 'train-arriveby', 'train-day', 'train-departure', 'train-destination', 'train-leaveat']
crosswoz_slot_list = ["����-��Ʊ", "����-����", "�͹�-����", "�Ƶ�-�۸�", "�Ƶ�-����", "����-����", "����-��ַ", "����-����ʱ��", "�͹�-Ӫҵʱ��", "�͹�-����", "�Ƶ�-����", "�Ƶ�-�ܱ߾���", "�Ƶ�-�Ƶ���ʩ-���ѷ���", "�Ƶ�-�Ƶ�����", "�͹�-�˾�����", "�͹�-�Ƽ���", "�Ƶ�-�Ƶ���ʩ", "�Ƶ�-�绰", "����-�绰", "�͹�-�ܱ߲͹�", "�͹�-�绰", "�͹�-none", "�͹�-��ַ", "�Ƶ�-�Ƶ���ʩ-���̷�", "�Ƶ�-��ַ", "����-�ܱ߾���", "����-�ܱ߾Ƶ�", "����-������", "����-Ŀ�ĵ�", "����-������", "����-Ŀ�ĵ�", "����-�ܱ߲͹�", "�Ƶ�-�ܱ߲͹�", "����-����", "�͹�-�ܱ߾���", "�͹�-�ܱ߾Ƶ�", "����-�����ظ�������վ", "����-Ŀ�ĵظ�������վ", "����-none", "�Ƶ�-�Ƶ���ʩ-��������", "�͹�-Դ����", "�Ƶ�-�Ƶ���ʩ-��ʽ����", "�Ƶ�-�Ƶ���ʩ-��վ����", "�Ƶ�-�Ƶ���ʩ-���ʳ�;�绰", "�Ƶ�-�Ƶ���ʩ-�����", "�Ƶ�-�Ƶ���ʩ-������", "�Ƶ�-Դ����", "�Ƶ�-none", "�Ƶ�-�Ƶ���ʩ-�������", "�Ƶ�-�Ƶ���ʩ-����С������", "�Ƶ�-�Ƶ���ʩ-�Ƶ�����ṩwifi", "�Ƶ�-�Ƶ���ʩ-ů��", "�Ƶ�-�Ƶ���ʩ-spa", "����-����", "����-Դ����", "�Ƶ�-�Ƶ���ʩ-����Ĵ�", "�Ƶ�-�Ƶ���ʩ-��ʽ����", "�Ƶ�-�Ƶ���ʩ-�ư�", "�Ƶ�-�Ƶ���ʩ-��ͷ���", "�Ƶ�-�Ƶ���ʩ-����", "�Ƶ�-�Ƶ���ʩ-�м�����ʩ", "�Ƶ�-�Ƶ���ʩ-������ڵ绰", "�Ƶ�-�Ƶ���ʩ-�Ӵ����", "�Ƶ�-�Ƶ���ʩ-���ַ����ṩwifi", "�Ƶ�-�Ƶ���ʩ-ϴ�·���", "�Ƶ�-�Ƶ���ʩ-�⳵", "�Ƶ�-�Ƶ���ʩ-��������Ͳ��ַ����ṩwifi", "�Ƶ�-�Ƶ���ʩ-24Сʱ��ˮ", "�Ƶ�-�Ƶ���ʩ-��Ȫ", "�Ƶ�-�Ƶ���ʩ-ɣ��", "�Ƶ�-�Ƶ���ʩ-�շ�ͣ��λ", "�Ƶ�-�ܱ߾Ƶ�", "�Ƶ�-�Ƶ���ʩ-�ӻ�����", "�Ƶ�-�Ƶ���ʩ-���з����ṩwifi", "�Ƶ�-�Ƶ���ʩ-������", "�Ƶ�-�Ƶ���ʩ-��ѹ��ڳ�;�绰", "�Ƶ�-�Ƶ���ʩ-������Ӿ��", "�Ƶ�-�Ƶ���ʩ-��ͷ������", "�Ƶ�-�Ƶ���ʩ-���������ṩwifi", "�Ƶ�-�Ƶ���ʩ-������Ӿ��"]


def format_history(context):
    history = []
    for i in range(len(context)):
        history.append(['sys' if i%2==1 else 'usr', context[i]])
    return history

def sentseg(sent):
    sent = sent.replace('\t', ' ')
    sent = ' '.join(sent.split())
    tmp = " ".join(jieba.cut(sent))
    return ' '.join(tmp.split())

def reformat_state(state):
    if 'belief_state' in state:
        state = state['belief_state']
    new_state = []
    for domain in state.keys():
        domain_data = state[domain]
        if 'semi' in domain_data:
            domain_data = domain_data['semi']
            for slot in domain_data.keys():
                val = domain_data[slot]
                if val is not None and val != '' and val != 'not mentioned':
                    new_state.append(domain + '-' + slot + '-' + val)
    # lower
    new_state = [item.lower() for item in new_state]
    return new_state

def reformat_state_crosswoz(state):
    if 'belief_state' in state:
        state = state['belief_state']
    new_state = []
    for domain in state.keys():
        domain_data = state[domain]
        for slot in domain_data.keys():
            if slot == 'selectedResults': continue
            val = domain_data[slot]
            if val is not None and val != '':
                new_state.append(domain + '-' + slot + '-' + val)
    return new_state

def compute_acc(gold, pred, slot_temp):
    # TODO: not mentioned in gold
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count

def evaluate_metrics(all_prediction, from_which, slot_temp):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for d, v in all_prediction.items():
        for t in range(len(v)):
            cv = v[t]
            if set(cv["turn_belief"]) == set(cv[from_which]):
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    turn_acc_score = turn_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
    return joint_acc_score, F1_score, turn_acc_score

if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if len(sys.argv) != 3:
        print("usage:")
        print("\t python evaluate.py dataset model")
        print("\t dataset=MultiWOZ, CrossWOZ")
        print("\t model=TRADE, mdbt, sumbt")
        sys.exit()

    ## init phase
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    if dataset_name == 'MultiWOZ':
        if model_name == 'TRADE':
            from convlab2.dst.trade.multiwoz.trade import MultiWOZTRADE
            model = MultiWOZTRADE()
        elif model_name == 'mdbt':
            from convlab2.dst.mdbt.multiwoz.dst import MultiWozMDBT
            model = MultiWozMDBT()
        elif model_name == 'sumbt':
            from convlab2.dst.sumbt.multiwoz.sumbt import SUMBTTracker
            model = SUMBTTracker()
        else:
            raise Exception("Available models: TRADE/mdbt/sumbt")

        ## load data
        from convlab2.util.dataloader.module_dataloader import AgentDSTDataloader
        from convlab2.util.dataloader.dataset_dataloader import MultiWOZDataloader
        dataloader = AgentDSTDataloader(dataset_dataloader=MultiWOZDataloader())
        data = dataloader.load_data(data_key='test')['test']
        context, golden_truth = data['context'], data['belief_state']
        all_predictions = {}
        test_set = []
        curr_sess = {}
        session_count = 0
        turn_count = 0
        is_start = True
        for i in tqdm(range(len(context))):
        # for i in tqdm(range(200)):  # for test
            if len(context[i]) == 0:
                turn_count = 0
                if is_start:
                    is_start = False
                else:  # save session
                    all_predictions[session_count] = copy.deepcopy(curr_sess)
                    session_count += 1
                    curr_sess = {}
            if golden_truth[i] == {}:
                continue
            # add turn
            x = context[i]
            y = golden_truth[i]
            model.init_session()
            model.state['history'] = format_history(context[i])
            pred = model.update(x[-1] if len(x) > 0 else '')
            curr_sess[turn_count] = {
                'turn_belief': reformat_state(y),
                'pred_bs_ptr': reformat_state(pred)
            }
            # print('golden: ', reformat_state(y))
            # print('pred :', reformat_state(pred))
            turn_count += 1
        # add last session
        if len(curr_sess) > 0:
            all_predictions[session_count] = copy.deepcopy(curr_sess)

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = evaluate_metrics(all_predictions, "pred_bs_ptr", multiwoz_slot_list)
        evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr,
                              "Joint F1": F1_score_ptr}
        print(evaluation_metrics)

    elif dataset_name == 'CrossWOZ':
        if model_name == 'TRADE':
            from convlab2.dst.trade.crosswoz.trade import CrossWOZTRADE
            model = CrossWOZTRADE()
        elif model_name == 'mdbt':
            pass
        elif model_name == 'sumbt':
            pass
        elif model_name == 'rule':
            pass
        else:
            raise Exception("Available models: TRADE")

        ## load data
        from convlab2.util.dataloader.module_dataloader import CrossWOZAgentDSTDataloader
        from convlab2.util.dataloader.dataset_dataloader import CrossWOZDataloader

        dataloader = CrossWOZAgentDSTDataloader(dataset_dataloader=CrossWOZDataloader())
        data = dataloader.load_data(data_key='test')['test']
        context, golden_truth = data['context'], data['sys_state_init']
        all_predictions = {}
        test_set = []
        curr_sess = {}
        session_count = 0
        turn_count = 0
        is_start = True
        for i in tqdm(range(len(context))):
        # for i in tqdm(range(10)):  # for test
            if len(context[i]) == 0:
                turn_count = 0
                if is_start:
                    is_start = False
                else:  # save session
                    all_predictions[session_count] = copy.deepcopy(curr_sess)
                    session_count += 1
                    curr_sess = {}
            if len(context[i]) % 2 == 0:
                continue
            # add turn
            x = context[i]
            y = golden_truth[i]
            # process y
            for domain in y.keys():
                domain_data = y[domain]
                for slot in domain_data.keys():
                    if slot == 'selectedResults': continue
                    val = domain_data[slot]
                    if val is not None and val != '':
                        val = sentseg(val)
                        y[domain][slot] = val
            model.init_session()
            model.state['history'] = format_history([sentseg(item) for item in context[i]])
            pred = model.update(x[-1] if len(x) > 0 else '')
            curr_sess[turn_count] = {
                'turn_belief': reformat_state_crosswoz(y),
                'pred_bs_ptr': reformat_state_crosswoz(pred)
            }
            turn_count += 1
        # add last session
        if len(curr_sess) > 0:
            all_predictions[session_count] = copy.deepcopy(curr_sess)

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = evaluate_metrics(all_predictions, "pred_bs_ptr",
                                                                                 crosswoz_slot_list)
        evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr,
                              "Joint F1": F1_score_ptr}
        print(evaluation_metrics)
