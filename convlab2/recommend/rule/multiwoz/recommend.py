# -*- coding: utf-8 -*-
# Copyright: WANG Hongru
# Time 2020 06 03 11:00 am
from convlab2.recommend.recommend import Recommend
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA
import os
import json

class RuleRecommend(Recommend):
    """Rule based Recommend which trivially provide options for user when the system does not understand what's user meaning.
    
    Attributes:
        
    
    """
    def __init__(self):
        Recommend.__init__(self)
        self.state = default_state()
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/value_dict.json')
        self.value_dict = json.load(open(path))
    
    def recommend(self, sf, status):
        """
        recommend belief_state, request_state
        :param sf, status
        :return options
        """
        if status == "unk":
            options = ["Hotel", "Attraction", "Restaurant"]
        return options
    

if __name__ == '__main__':
    recommend = RuleRecommend()
    