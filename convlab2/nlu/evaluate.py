"""
Evaluate NLU models on specified dataset
Metric: dataset level Precision/Recall/F1
Usage: python evaluate.py [CrossWOZ|MultiWOZ|Camrest] [BERTNLU|MILU|SVMNLU] [usr|sys|all]
"""

import json
import random
import sys
import zipfile
import numpy
import torch
from pprint import pprint
from tqdm import tqdm


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        predicts = [[x[0], x[1], x[2], x[3].lower()] for x in predicts]
        labels = item['golden']
        labels = [[x[0], x[1], x[2], x[3].lower()] for x in labels]
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                FP += 1
        for ele in labels:
            if ele not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1


if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if len(sys.argv) != 4:
        print("usage:")
        print("\t python evaluate.py dataset model role")
        print("\t dataset=MultiWOZ, CrossWOZ, or Camrest")
        print("\t model=BERTNLU, MILU, or SVMNLU")
        print("\t role=usr, sys or all")
        sys.exit()
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    if dataset_name == 'MultiWOZ':
        if model_name == 'MILU':
            from convlab2.nlu.milu.multiwoz import MILU
            model = MILU()
        elif model_name == 'SVMNLU':
            from convlab2.nlu.svm.multiwoz import SVMNLU
            model = SVMNLU()
        elif model_name == 'BERTNLU':
            from convlab2.nlu.jointBERT.multiwoz import BERTNLU
            model = BERTNLU()
        else:
            raise Exception("Available models: MILU, SVMNLU, BERTNLU")

        from convlab2.util.dataloader.module_dataloader import MultiTurnNLUDataloader
        from convlab2.util.dataloader.dataset_dataloader import MultiWOZDataloader
        dataloader = MultiTurnNLUDataloader(dataset_dataloader=MultiWOZDataloader())
        data = dataloader.load_data(data_key='test', role=sys.argv[3])['test']
        predict_golden = []
        for i in tqdm(range(len(data['utterance']))):
            predict = model.predict(utterance=data['utterance'][i],
                                    context=data['context'][i])
            label = data['dialog_act'][i]
            predict_golden.append({
                'predict': predict,
                'golden': label
            })

        precision, recall, F1 = calculateF1(predict_golden)
        print('Model {} on {} {} sentences:'.format(model_name, dataset_name, len(predict_golden)))
        print('\t Precision: %.2f' % (100 * precision))
        print('\t Recall: %.2f' % (100 * recall))
        print('\t F1: %.2f' % (100 * F1))

    else:
        raise Exception("currently supported dataset: MultiWOZ")
