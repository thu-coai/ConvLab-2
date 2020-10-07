# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from collections import namedtuple
import random


class DiscretePolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super(DiscretePolicy, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, a_dim))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [1]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.softmax(a_weights, 0)

        # randomly sample from normal distribution, whose mean and variance come from policy network.
        # [a_dim] => [1]
        a = a_probs.multinomial(1) if sample else a_probs.argmax(0, True)

        return a

    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, 1]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.softmax(a_weights, -1)

        # [b, a_dim] => [b, 1]
        trg_a_probs = a_probs.gather(-1, a)
        log_prob = torch.log(trg_a_probs)

        return log_prob
 
class EpsilonGreedyPolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim, epsilon_spec={'start': 0.1, 'end': 0.0, 'end_epoch': 200}):
        super(EpsilonGreedyPolicy, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, a_dim))

        self.epsilon = epsilon_spec['start']
        self.start = epsilon_spec['start']
        self.end = epsilon_spec['end']
        self.end_epoch = epsilon_spec['end_epoch']
        self.a_dim = a_dim

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, is_train=True):
        """
        :param s: [s_dim]
        :return: [1]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]

        if is_train:
            if self.epsilon > np.random.rand():
                # select a random action
                a = torch.randint(self.a_dim, (1, ))
            else:
                a = self._greedy_action(s)
        else:
            a = self._greedy_action(s)

        # transforms action index to a vector action (one-hot encoding)
        a_vec = torch.zeros(self.a_dim)
        a_vec[a] = 1.

        return a_vec

    def update_epsilon(self, epoch):
        # Linear decay
        a = -float(self.start - self.end) / self.end_epoch
        b = float(self.start)
        self.epsilon = max(self.end, a * float(epoch) + b)

    def _greedy_action(self, s):
        """
        Select a greedy action
        """
        a_weights = self.forward(s)
        return a_weights.argmax(0, True)

class MultiDiscretePolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super(MultiDiscretePolicy, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, a_dim))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights
    
    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)
        
        # [a_dim] => [a_dim, 2]
        a_probs = a_probs.unsqueeze(1)
        a_probs = torch.cat([1-a_probs, a_probs], 1)
        a_probs = torch.clamp(a_probs, 1e-10, 1 - 1e-10)
        
        # [a_dim, 2] => [a_dim]
        a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)
        
        return a
    
    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)
        
        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(-1)
        a_probs = torch.cat([1-a_probs, a_probs], -1)
        
        # [b, a_dim, 2] => [b, a_dim]
        trg_a_probs = a_probs.gather(-1, a.unsqueeze(-1).long()).squeeze(-1)
        log_prob = torch.log(trg_a_probs)
        
        return log_prob.sum(-1, keepdim=True)
        

class ContinuousPolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super(ContinuousPolicy, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU())
        self.net_mean = nn.Linear(h_dim, a_dim)
        self.net_std = nn.Linear(h_dim, a_dim)

    def forward(self, s):
        # [b, s_dim] => [b, h_dim]
        h = self.net(s)

        # [b, h_dim] => [b, a_dim]
        a_mean = self.net_mean(h)
        a_log_std = self.net_std(h)

        return a_mean, a_log_std

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action mean and log_std
        # [s_dim] => [a_dim]
        a_mean, a_log_std = self.forward(s)

        # randomly sample from normal distribution, whose mean and variance come from policy network.
        # [a_dim]
        a = torch.normal(a_mean, a_log_std.exp()) if sample else a_mean

        return a

    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        def normal_log_density(x, mean, log_std):
            """
            x ~ N(mean, std)
            this function will return log(prob(x)) while x belongs to guassian distrition(mean, std)
            :param x:       [b, a_dim]
            :param mean:    [b, a_dim]
            :param log_std: [b, a_dim]
            :return:        [b, 1]
            """
            std = log_std.exp()
            var = std.pow(2)
            log_density = - (x - mean).pow(2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std
        
            return log_density.sum(-1, keepdim=True)
        
        # forward to get action mean and log_std
        # [b, s_dim] => [b, a_dim]
        a_mean, a_log_std = self.forward(s)

        # [b, a_dim] => [b, 1]
        log_prob = normal_log_density(a, a_mean, a_log_std)

        return log_prob
    
    
class Value(nn.Module):
    def __init__(self, s_dim, hv_dim):
        super(Value, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(hv_dim, hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(hv_dim, 1))

    def forward(self, s):
        """
        :param s: [b, s_dim]
        :return:  [b, 1]
        """
        value = self.net(s)

        return value

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))

class Memory(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

class MemoryReplay(object):
    """
        The difference to class Memory is that MemoryReplay has a limited size.
        It is mainly used for off-policy algorithms.
    """
    def __init__(self, max_size):
        self.memory = []
        self.index = 0
        self.max_size = max_size

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.index] = Transition(*args)
        self.index = (self.index + 1) % self.max_size
        

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        for transition in new_memory.memory:
            if len(self.memory) < self.max_size:
                self.memory.append(None)
            self.memory[self.index] = transition
            self.index = (self.index + 1) % self.max_size

    def reset(self):
        self.memory = []
        self.index = 0

    def __len__(self):
        return len(self.memory)        
