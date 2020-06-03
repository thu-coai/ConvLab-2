# -*- coding: utf-8 -*-
import torch
import os
import json
from convlab2.policy.mle.mle import MLEAbstract
from convlab2.policy.rlmodule import MultiDiscretePolicy
from convlab2.policy.vector.vector_camrest import CamrestVector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "mle_policy_camrest.zip")

class MLE(MLEAbstract):
    
    def __init__(self,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 model_file='https://convlab.blob.core.windows.net/convlab-2/mle_policy_camrest.zip'):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        
        voc_file = os.path.join(root_dir, 'data/camrest/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/camrest/usr_da_voc.txt')
        self.vector = CamrestVector(voc_file, voc_opp_file)
               
        self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)
        
        self.load(archive_file, model_file, cfg['load'])
