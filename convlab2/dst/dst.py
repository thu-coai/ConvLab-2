"""Dialog State Tracker Interface"""
import copy
from abc import abstractmethod

from convlab2.util.module import Module


class DST(Module):
    """DST module interface."""

    def update(self, action):
        """ Update the internal dialog state variable.

        Args:
            action (str or list of list):
                The type is str when DST is word-level (such as NBT), and list of list when it is DA-level.
        Returns:
            new_state (dict):
                Updated dialog state, with the same form of previous state.
        """
        pass

    def update_turn(self, sys_utt, user_utt):
        """ Update the internal dialog state variable with .

        Args:
            sys_utt (str):
                system utterance of current turn, set to `None` for the first turn
            user_utt (str):
                user utterance of current turn
        Returns:
            new_state (dict):
                Updated dialog state, with the same form of previous state.
        """
        pass

    def to_cache(self, *args, **kwargs):
        return copy.deepcopy(self.state)

    def from_cache(self, *args, **kwargs):
        self.state = copy.deepcopy(args[0])
