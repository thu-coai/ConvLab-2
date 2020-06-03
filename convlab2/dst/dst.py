"""Dialog State Tracker Interface"""
from convlab2.util.module import Module
import copy


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

    def to_cache(self, *args, **kwargs):
        return copy.deepcopy(self.state)

    def from_cache(self, *args, **kwargs):
        self.state = copy.deepcopy(args[0])
