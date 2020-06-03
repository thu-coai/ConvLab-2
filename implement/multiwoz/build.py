# -*- coding: utf-8 -*-
# Copyright Wang Hongru
# Time: 2020 05 27 15pm
import sys
sys.path.insert(0, "/Users/wanghongru/Documents/gitProject/chatbot/platform/ConvLab-2/")

from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import random
import torch
import numpy as np

# build different components for system agents
sys_nlu = BERTNLU()
sys_dst = RuleDST()
sys_policy = RulePolicy()
sys_nlg = TemplateNLG(is_user=False)
sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name="sys")

# chat with the agent using its response functions
print(sys_agent.response("I want to find a moderate hotel."))
print(sys_agent.response("Which type of hotel is it?"))
print(sys_agent.response("Ok, where is its address?"))
print(sys_agent.response("Thank u!"))
print(sys_agent.response("Try to find me a Chinese restaurant in south area."))

# build a simulator to chat with the agent and evaluate
user_nlu = BERTNLU()
user_dst = None # here we choose Agenda policy for the simulator
user_policy = RulePolicy(character='usr')
user_nlg = TemplateNLG(is_user=True)
user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

# evaluate the model first init the goal and evaluate based on the model finish the goal
evaluator = MultiWozEvaluator()
sess = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=evaluator)
def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)

set_seed(20200131)

sys_response = ''
sess.init_session()
print('init goal:')
pprint(sess.evaluator.goal)
print('-'*50)
for i in range(20):
    sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
    print('user:', user_response)
    print('sys:', sys_response)
    print()
    if session_over is True:
        break
print('task success:', sess.evaluator.task_success())
print('book rate:', sess.evaluator.book_rate())
print('inform precision/recall/f1:', sess.evaluator.inform_F1())
print('-'*50)
print('final goal:')
pprint(sess.evaluator.goal)
print('='*100)





