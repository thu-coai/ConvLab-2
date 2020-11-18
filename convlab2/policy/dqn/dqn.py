# -*- coding: utf-8 -*-
import torch
from torch import optim
from torch import nn
import numpy as np
import logging
import os
import json
import copy
from convlab2.policy.policy import Policy
from convlab2.policy.rlmodule import EpsilonGreedyPolicy, MemoryReplay
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.policy.rule.multiwoz.rule_based_multiwoz_bot import RuleBasedMultiwozBot
from convlab2.util.file_util import cached_path
import zipfile
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(Policy):

    def __init__(self, is_train=False, dataset='Multiwoz'):

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        self.save_per_epoch = cfg['save_per_epoch']
        self.training_iter = cfg['training_iter']
        self.training_batch_iter = cfg['training_batch_iter']
        self.batch_size = cfg['batch_size']
        self.epsilon = cfg['epsilon_spec']['start']
        self.rule_bot = RuleBasedMultiwozBot()
        self.gamma = cfg['gamma']
        self.is_train = is_train
        if is_train:
            init_logging_handler(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['log_dir']))

        # construct multiwoz vector
        if dataset == 'Multiwoz':
            voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
            self.vector = MultiWozVector(voc_file, voc_opp_file, composite_actions=True, vocab_size=cfg['vocab_size'])

        #replay memory
        self.memory = MemoryReplay(cfg['memory_size'])

        self.net = EpsilonGreedyPolicy(self.vector.state_dim, cfg['hv_dim'], self.vector.da_dim, cfg['epsilon_spec']).to(device=DEVICE)
        self.target_net = copy.deepcopy(self.net)

        self.online_net = self.target_net
        self.eval_net = self.target_net

        if is_train:
            self.net_optim = optim.Adam(self.net.parameters(), lr=cfg['lr'])

        self.loss_fn = nn.MSELoss()

    def update_memory(self, sample):
        self.memory.reset()
        self.memory.append(sample)
        
    def predict(self, state, warm_up=False):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        if warm_up:
            action = self.rule_action(state)
            state['system_action'] = action
        else:
            s_vec = torch.Tensor(self.vector.state_vectorize(state))
            a = self.net.select_action(s_vec.to(device=DEVICE), is_train=self.is_train)
            action = self.vector.action_devectorize(a.numpy())
            state['system_action'] = action
        return action
    
    def rule_action(self, state):
        if self.epsilon > np.random.rand():
            a = torch.randint(self.vector.da_dim, (1, ))
            # transforms action index to a vector action (one-hot encoding)
            a_vec = torch.zeros(self.vector.da_dim)
            a_vec[a] = 1.
            action = self.vector.action_devectorize(a_vec.numpy())
        else:
            # rule-based warm up
            action = self.rule_bot.predict(state)
        
        return action

    def init_session(self):
        """
        Restore after one session
        """
        self.memory.reset()
    
    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
        next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)

        q_preds = self.net(s)
        with torch.no_grad():
            # Use online_net to select actions in next state
            online_next_q_preds = self.online_net(next_s)
            # Use eval_net to calculate next_q_preds for actions chosen by online_net
            next_q_preds = self.eval_net(next_s)
        act_q_preds = q_preds.gather(-1, a.argmax(-1).long().unsqueeze(-1)).squeeze(-1)
        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
        max_q_targets = r + self.gamma * mask * max_next_q_preds
        
        
        q_loss = self.loss_fn(act_q_preds, max_q_targets)

        return q_loss
    
    def update(self, epoch):
        total_loss = 0.
        for i in range(self.training_iter):
            round_loss = 0.
            # 1. batch a sample from memory
            batch = self.memory.get_batch(batch_size=self.batch_size)
            
            for _ in range(self.training_batch_iter):
                # 2. calculate the Q loss
                loss = self.calc_q_loss(batch)

                # 3. make a optimization step
                self.net_optim.zero_grad()
                loss.backward()
                self.net_optim.step()

                round_loss += loss.item()

            logging.debug('<<dialog policy dqn>> epoch {}, iteration {}, loss {}'.format(epoch, i, round_loss / self.training_batch_iter))
            total_loss += round_loss
        total_loss /= (self.training_batch_iter * self.training_iter)
        logging.debug('<<dialog policy dqn>> epoch {}, total_loss {}'.format(epoch, total_loss))

        # update the epsilon value
        self.net.update_epsilon(epoch)

        # update the target network
        self.target_net.load_state_dict(self.net.state_dict())
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
    
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.net.state_dict(), directory + '/' + str(epoch) + '_dqn.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))
    
    def load(self, filename):
        dqn_mdl_candidates = [
            filename + '_dqn.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_dqn.pol.mdl'),
        ]

        for dqn_mdl in dqn_mdl_candidates:
            if os.path.exists(dqn_mdl):
                self.net.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
                self.target_net.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(dqn_mdl))
                break

    @classmethod
    def from_pretrained(cls,
                        archive_file="",
                        model_file="https://convlab.blob.core.windows.net/convlab-2/dqn_policy_multiwoz.zip",
                        is_train=False,
                        dataset='Multiwoz'):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        model = cls(is_train=is_train, dataset=dataset)
        model.load(cfg['load'])
        return model