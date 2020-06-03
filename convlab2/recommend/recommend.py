# -*- coding: utf-8 -*-
# Copyright: WANG Hongru
# Time: 2020 05 29 13:40 pm
from convlab2.util.module import Module

class Recommend(Module):
    """Recommend modeule inference"""
    def recommend(self, sf, status):
        """Recommend or provide some options for user.
        
        Args:
            sf: semantic frame,
            status: unknown, not sure

        Returns:
            options (list of str): options
        """
        return []