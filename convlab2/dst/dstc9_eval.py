import random
from argparse import ArgumentParser

import numpy as np
import torch

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=23333)
parser.add_argument('--subtask', type=str, required=True, choices=['multiwoz', 'crosswoz'])
args = parser.parse_args()

# make your model's behavior deterministic
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f'fix random seed: {seed}')

subtask = args.subtask

if __name__ == '__main__':
    pass