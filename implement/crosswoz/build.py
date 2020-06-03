# -*- coding: utf-8 -*-
# Copyright WANG Hongru
# Time: 2020 06 01 20pm
import sys
sys.path.insert(0, "/Users/wanghongru/Documents/gitProject/chatbot/platform/ConvLab-2/")

from convlab2.nlu.jointBERT.crosswoz import BERTNLU
from convlab2.dst.rule.crosswoz import RuleDST
from convlab2.policy.mle.crosswoz import MLE
from convlab2.nlg.template.crosswoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent, BiSession
from pprint import pprint
import random
import torch
import numpy as np

# build different components for system agents
sys_nlu = BERTNLU()
sys_dst = RuleDST()
sys_policy = MLE()
sys_nlg = TemplateNLG(is_user=False)

sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name="sys")

# chat with the agent using its response functions
for i in range(5):
    user = input("请输入:")
    print(sys_agent.response(input))
