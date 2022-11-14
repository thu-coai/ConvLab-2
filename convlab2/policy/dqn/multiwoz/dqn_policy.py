from convlab2.policy.dqn import DQN
import os
import json

class DQNPolicy(DQN):
    def __init__(self,
                is_train=False,
                dataset="Multiwoz",
                archive_file="",
                model_file="https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/dqn_policy_multiwoz.zip"):
        super().__init__(is_train=is_train, dataset=dataset)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        self.load(cfg['load'])