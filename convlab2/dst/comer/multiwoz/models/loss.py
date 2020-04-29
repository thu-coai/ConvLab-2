import torch
import torch.nn as nn
from convlab2.dst.comer.multiwoz import dict
from torch.autograd import Variable


def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    #crit = nn.CrossEntropyLoss(weight, size_average=False)
    crit = nn.NLLLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit
