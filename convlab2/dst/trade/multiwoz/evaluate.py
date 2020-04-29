"""
Evaluate NLU models on specified dataset
Usage: python evaluate.py [MultiWOZ|CrossWOZ] [TRADE|mdbt|sumbt|rule]
"""
import random
import numpy
import torch
from convlab2.dst.trade.multiwoz.trade import MultiWOZTRADE

if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    model = MultiWOZTRADE('model/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.3591')
    model.evaluate()