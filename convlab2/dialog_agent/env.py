# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:27:34 2019

@author: truthless
"""

class Environment():
    
    def __init__(self, sys_nlg, usr, sys_nlu, sys_dst, evaluator=None):
        self.sys_nlg = sys_nlg
        self.usr = usr
        self.sys_nlu = sys_nlu
        self.sys_dst = sys_dst
        self.evaluator = evaluator
        
    def reset(self):
        self.usr.init_session()
        self.sys_dst.init_session()
        if self.evaluator:
            self.evaluator.add_goal(self.usr.policy.get_goal())
        s, r, t = self.step([])
        return self.sys_dst.state
        
    def step(self, action):
        model_response = self.sys_nlg.generate(action) if self.sys_nlg else action
        observation = self.usr.response(model_response)
        if self.evaluator:
            self.evaluator.add_sys_da(self.usr.get_in_da())
            self.evaluator.add_usr_da(self.usr.get_out_da())
        dialog_act = self.sys_nlu.predict(observation) if self.sys_nlu else observation
        self.sys_dst.state['user_action'] = dialog_act
        state = self.sys_dst.update(dialog_act)
        
        if self.evaluator:
            if self.evaluator.task_success():
                reward = 40
            elif self.evaluator.cur_domain and self.evaluator.domain_success(self.evaluator.cur_domain):
                reward = 5
            else:
                reward = -1
        else:
            reward = self.usr.get_reward()
        terminated = self.usr.is_terminated()
        
        return state, reward, terminated
