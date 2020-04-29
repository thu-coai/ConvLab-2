"""
Evaluate NLU models on specified dataset
Metric: dataset level Precision/Recall/F1
Usage: python evaluate.py [MultiWOZ] [SCLSTM|TemplateNLG] [usr|sys]
"""

import json
import random
import sys
import zipfile
import numpy
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pprint import pprint
from tqdm import tqdm


def get_bleu4(dialog_acts, golden_utts, gen_utts):
    das2utts = {}
    for das, utt, gen in zip(dialog_acts, golden_utts, gen_utts):
        utt = utt.lower()
        gen = gen.lower()
        for da in das:
            act, domain, s, v = da
            if act == 'Request' or domain == 'general':
                continue
            else:
                if s == 'Internet' or s == 'Parking' or s == 'none' or v == 'none':
                    continue
                else:
                    v = v.lower()
                    if (' ' + v in utt) or (v + ' ' in utt):
                        utt = utt.replace(v, '{}-{}'.format(act + '-' + domain, s), 1)
                    if (' ' + v in gen) or (v + ' ' in gen):
                        gen = gen.replace(v, '{}-{}'.format(act + '-' + domain, s), 1)
        hash_key = ''
        for da in sorted(das, key=lambda x: x[0] + x[1] + x[2]):
            hash_key += '-'.join(da[:-1]) + ';'
        das2utts.setdefault(hash_key, {'refs': [], 'gens': []})
        das2utts[hash_key]['refs'].append(utt)
        das2utts[hash_key]['gens'].append(gen)
    # pprint(das2utts)
    refs, gens = [], []
    for das in das2utts.keys():
        for gen in das2utts[das]['gens']:
            refs.append([s.split() for s in das2utts[das]['refs']])
            gens.append(gen.split())
    bleu = corpus_bleu(refs, gens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    return bleu


if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if len(sys.argv) != 4:
        print("usage:")
        print("\t python evaluate.py dataset model role")
        print("\t dataset=MultiWOZ, CrossWOZ, or Camrest")
        print("\t model=SCLSTM, or TemplateNLG")
        print("\t role=usr/sys")
        sys.exit()
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    role = sys.argv[3]
    if dataset_name == 'MultiWOZ':
        if model_name == 'SCLSTM':
            from convlab2.nlg.sclstm.multiwoz import SCLSTM
            if role == 'usr':
                model = SCLSTM(is_user=True, use_cuda=True)
            elif role == 'sys':
                model = SCLSTM(is_user=False, use_cuda=True)
        elif model_name == 'TemplateNLG':
            from convlab2.nlg.template.multiwoz import TemplateNLG
            if role == 'usr':
                model = TemplateNLG(is_user=True)
            elif role == 'sys':
                model = TemplateNLG(is_user=False)
        else:
            raise Exception("Available models: SCLSTM, TEMPLATE")

        from convlab2.util.dataloader.module_dataloader import SingleTurnNLGDataloader
        from convlab2.util.dataloader.dataset_dataloader import MultiWOZDataloader
        dataloader = SingleTurnNLGDataloader(dataset_dataloader=MultiWOZDataloader())
        data = dataloader.load_data(data_key='test', role=role)['test']

        dialog_acts = []
        golden_utts = []
        gen_utts = []
        gen_slots = []

        sen_num = 0

        for i in tqdm(range(len(data['utterance']))):
            dialog_acts.append(data['dialog_act'][i])
            golden_utts.append(data['utterance'][i])
            gen_utts.append(model.generate(data['dialog_act'][i]))

        bleu4 = get_bleu4(dialog_acts, golden_utts, gen_utts)

        print("Calculate bleu-4")
        print("BLEU-4: %.4f" % bleu4)

        print('Model on {} sentences role={}'.format(len(data['utterance']), role))

    else:
        raise Exception("currently supported dataset: MultiWOZ")
