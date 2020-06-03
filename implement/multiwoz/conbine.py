# -*- coding: utf-8 -*-
# Copyright Wang Hongru
# 2020 05 27 21:00 pm

# available model in Convlab2

# NLU: BERTNLU MILU SVMNLU
# DST: RuleDST
# word-DST: MDBT(set sys_nlu to None) TRADE SUMBT
# Policy: RulePolicy Imitation REINFORCE PPO GDPL
# word-policy: MDRG HDSA LaRL(set sys_nlg to None)
# NLG: Template SCLSTM
# End2end Sequicity DAMD RNN_rollout(directly used as sys_agent)
# Simulator policy: Agenda VHUS(for user_policy)

# available NLU models
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.nlu.jointBert.multiwoz import BERTNLU
from convalb2.nlu.milu.multiwoz import MILU
# available DST models
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.dst.mdbt.multiwoz import MDBT
from convlab2.dst.sumbt.multiwoz import SUMBT
from convlab2.dst.trade.multiwoz import TRADE
# available Policy model
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo.multiwoz import PPOPolicy
from convlab2.policy.pg.multiwoz import PGPolicy
from convlab2.policy.mle.multiwoz import MLEPolicy
from convlab2.policy.gdpl.multiwoz import GDPLPolicy
from convlab2.policy.vhus.multiwoz import UserPolicyVHUS
from convlab2.policy.mdrg.multiwoz import MDRGWordPolicy
from convlab2.policy.hdsa.multiwoz import HDSA
from convlab2.policy.larl.multiwoz import LaRL
# available NLG models
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.nlg.sclstm.multiwoz import SCLSTM
# available E2E models
from convlab2.e2e.sequicity.multiwoz import Sequicity
from convlab2.e2e.damd.multiwoz import Damd


# NLU + RuleDST or Word-DST
sys_nlu = BERTNLU()
# sys_nlu = MILU()
# sys_nlu = SVMNLU()
sys_dst = RuleDST()

# or Word-DST
# sys_nlu = None
# sys_dst = SUMBT()
# sys_dst = TRADE()
# sys_dst = MDBT()

# Policy+NLG:
sys_policy = RulePolicy()
# sys_policy = PPOPolicy()
# sys_policy = PGPolicy()
# sys_policy = MLEPolicy()
# sys_policy = GDPLPolicy()
sys_nlg = TemplateNLG(is_user=False)
# sys_nlg = SCLSTM(is_user=False)

# or Word-Policy:
# sys_policy = LaRL()
# sys_policy = HDSA()
# sys_policy = MDRGWordPolicy()
# sys_nlg = None

# assemble the system pipeline
sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')

# or directly use end-to-end systems
# sys_agent = Sequicity()
# sys_agent = Damd()

user_nlu = BERTNLU()
# user_nlu = MILU()
# user_nlu = SVMNLU()
user_dst = None
user_policy = RulePolicy(character='usr')
# user_policy = UserPolicyVHUS(load_from_zip=True)
user_nlg = TemplateNLG(is_user=True)
# user_nlg = SCLSTM(is_user=True)
user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')


# use analysis tool to diagnose the system
from convlab2.util.analysis_tool.analyzer import Analyzer

# if sys_nlu!=None, set use_nlu=True to collect more information
analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

set_seed(20200131)
analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='sys_agent', total_dialog=100)









