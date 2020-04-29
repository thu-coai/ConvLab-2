"""Natural Language Understanding Interface."""
from convlab2.util.module import Module


class NLU(Module):
    """NLU module interface."""

    def predict(self, utterance, context=list()):
        """Predict the dialog act of a natural language utterance.
        
        Args:
            utterance (str):
                A natural language utterance.
            context (list of str):
                Previous utterances.

        Returns:
            action (list of list):
                The dialog act of utterance.
        """
        return []
