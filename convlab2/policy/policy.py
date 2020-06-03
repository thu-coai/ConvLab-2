"""Policy Interface"""
from convlab2.util.module import Module


class Policy(Module):
    """Policy module interface."""

    def predict(self, state):
        """Predict the next agent action given dialog state.

        Args:
            state (dict or list of list):
                when the policy takes dialogue state as input, the type is dict.
                else when the policy takes dialogue act as input, the type is list of list.
        Returns:
            action (list of list or str):
                when the policy outputs dialogue act, the type is list of list.
                else when the policy outputs utterance directly, the type is str.
        """
        return []
