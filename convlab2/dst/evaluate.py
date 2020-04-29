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
crosswoz_slot_list = ["景点-门票", "景点-评分", "餐馆-名称", "酒店-价格", "酒店-评分", "景点-名称", "景点-地址", "景点-游玩时间", "餐馆-营业时间", "餐馆-评分", "酒店-名称", "酒店-周边景点", "酒店-酒店设施-叫醒服务", "酒店-酒店类型", "餐馆-人均消费", "餐馆-推荐菜", "酒店-酒店设施", "酒店-电话", "景点-电话", "餐馆-周边餐馆", "餐馆-电话", "餐馆-none", "餐馆-地址", "酒店-酒店设施-无烟房", "酒店-地址", "景点-周边景点", "景点-周边酒店", "出租-出发地", "出租-目的地", "地铁-出发地", "地铁-目的地", "景点-周边餐馆", "酒店-周边餐馆", "出租-车型", "餐馆-周边景点", "餐馆-周边酒店", "地铁-出发地附近地铁站", "地铁-目的地附近地铁站", "景点-none", "酒店-酒店设施-商务中心", "餐馆-源领域", "酒店-酒店设施-中式餐厅", "酒店-酒店设施-接站服务", "酒店-酒店设施-国际长途电话", "酒店-酒店设施-吹风机", "酒店-酒店设施-会议室", "酒店-源领域", "酒店-none", "酒店-酒店设施-宽带上网", "酒店-酒店设施-看护小孩服务", "酒店-酒店设施-酒店各处提供wifi", "酒店-酒店设施-暖气", "酒店-酒店设施-spa", "出租-车牌", "景点-源领域", "酒店-酒店设施-行李寄存", "酒店-酒店设施-西式餐厅", "酒店-酒店设施-酒吧", "酒店-酒店设施-早餐服务", "酒店-酒店设施-健身房", "酒店-酒店设施-残疾人设施", "酒店-酒店设施-免费市内电话", "酒店-酒店设施-接待外宾", "酒店-酒店设施-部分房间提供wifi", "酒店-酒店设施-洗衣服务", "酒店-酒店设施-租车", "酒店-酒店设施-公共区域和部分房间提供wifi", "酒店-酒店设施-24小时热水", "酒店-酒店设施-温泉", "酒店-酒店设施-桑拿", "酒店-酒店设施-收费停车位", "酒店-周边酒店", "酒店-酒店设施-接机服务", "酒店-酒店设施-所有房间提供wifi", "酒店-酒店设施-棋牌室", "酒店-酒店设施-免费国内长途电话", "酒店-酒店设施-室内游泳池", "酒店-酒店设施-早餐服务免费", "酒店-酒店设施-公共区域提供wifi", "酒店-酒店设施-室外游泳池"]


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
