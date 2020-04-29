# Add New Model

- [Add NLU model](#Add-NLU-model)
- [Add DST model](#Add-DST-model)
- [Add Policy model](#Add-Policy-model)
- [Add NLG model](#Add-NLG-model)
- [Add End2End model](#Add-End2End-model)

## Add NLU model

we will take BERTNLU as an example to show how to add new NLU model to **ConvLab-2**.

### NLU interface

To make the new model consistent with **ConvLab-2**, we should follow the NLU interface definition in `convlab2/nlu/nlu.py`. The key function is `predict` which takes an utterance (str) and context (list of str) as inputs and return the dialog act. The format of dialog act is depended on specific dataset. For MultiWOZ dataset, it looks like `[["Inform", "Restaurant", "Food", "brazilian"], ["Inform", "Restaurant", "Area","north"]]`.


```python
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
```

### Add New Model

In order to add new Model to **ConvLab-2**, we should inherit the `NLU` class above. Here is a piece from BERTNLU. 


```python
class BERTNLU(NLU):
    def __init__(self, mode, model_file):
        ## model initialization here, feel free to change the arguments
        self.model = BertNLU()

    def predict(self, utterance, context):
        return self.model.predict(utterance, context)
```



## Add DST model

we will take RuleDST as an example to show how to add new DST model to **ConvLab-2**.

### DST interface

To make the new model consistent with **ConvLab-2**, we should follow the DST interface definition in `convlab2/dst/dst.py`. The key function is `update` which takes dialog_act or user utterance as input, update the `state` attribute and return it. The state format is depended on specific dataset. For MultiWOZ dataset, it is defined in `convlab2/util/multiwoz/state.py`. Note that the DST could take dialogue history from its attribute `state` as input too.


```python
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
```

### Add New Model

In order to add new Model to **ConvLab-2**, we should inherit the `DST` class above. 


```python
from convlab2.util.multiwoz.state import default_state

class RuleDST(DST):
    def __init__(self):
        ## model initialization here, feel free to change the arguments
        self.state = default_state()

    def update(self, user_act=None):
        # modify self.state
        return copy.deepcopy(self.state)
    
    def init_session(self):
        """Initialize ``self.state`` with a default state, which ``tatk.util.camrest.state.default_state`` returns."""
        self.state = default_state()
```



## Add Policy Model

we will take RulePolicy as an example to show how to add new Policy model to **ConvLab-2**.

### Policy interface

To make the new model consistent with **ConvLab-2**, we should follow the Policy interface definition in `convlab2/policy/policy.py`. The key function is `predict` which takes state(dict) as input and outputs dialog act. The state format is depended on specific dataset. For MultiWOZ dataset, it is defined in `convlab2/util/multiwoz/state.py`. 


```python
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
```

### Add New Model

In order to add new Model to **ConvLab-2**, we should inherit the `Policy` class above. 


```python
class RulePolicy(Policy):
    def __init__(self, is_train=False, character='sys'):
        ## model initialization here, feel free to change the arguments
        self.policy = UserPolicyAgendaMultiWoz()
        
    def predict(self, state):
        action = self.policy.predict(state)
        return action

    def init_session(self):
        pass
```



## Add NLG Model

we will take TemplateNLG as an example to show how to add new NLG model to **ConvLab-2**.

### NLG interface

To make the new model consistent with **ConvLab-2**, we should follow the NLG interface definition in `convlab2/nlg/nlg.py`. The key function is `generate` which takes the dialog act as input and return an utterance(str). The dialog act format is depended on specific dataset. For MultiWOZ dataset, it looks like `[["Inform", "Restaurant", "Food", "brazilian"], ["Inform", "Restaurant", "Area","north"]]`.


```python
class NLG(Module):
    """NLG module interface."""

    def generate(self, action):
        """Generate a natural language utterance conditioned on the dialog act.
        
        Args:
            action (list of list):
                The dialog action produced by dialog policy module, which is in dialog act format.
        Returns:
            utterance (str):
                A natural langauge utterance.
        """
        return ''
```

### Add New Model

In order to add new Model to **ConvLab-2**, we should inherit the `NLG` class above. 


```python
class TemplateNLG(NLG):
    def __init__(self, is_user, mode="manual"):
        ## model initialization here, feel free to change the arguments
        self.template = Template(is_user)

    def generate(self, dialog_acts):
        return self.template.generate(dialog_acts)
```



## Add End2End Model

we will take Sequicity as an example to show how to add new End-to-End model to **ConvLab-2**.

### End2End interface

To make the new model consistent with **ConvLab-2**, we should follow the `Agent` interface definition in `convlab2/dialog_agent/agent.py`. The key function is `response` which takes an utterance(str) as input and return an utterance(str). 


```python
class Agent(ABC):
    """Interface for dialog agent classes."""
    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def response(self, observation):
        """Generate agent response given user input.

        The data type of input and response can be either str or list of tuples, condition on the form of agent.

        Example:
            If the agent is a pipeline agent with NLU, DST and Policy, then type(input) == str and type(response) == list of tuples.
        Args:
            observation (str or list of tuples):
                The input to the agent.
        Returns:
            response (str or list of tuples):
                The response generated by the agent.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Reset the class variables to prepare for a new session."""
        pass
```

### Add New Model

In order to add new Model to **ConvLab-2**, we should inherit the `Agent` class above. 


```python
class Sequicity(Agent):
    def __init__(self, model_file=None):
        self.init_session()
        
    def response(self, usr):
        return self.generate(usr)
        
    def init_session(self):
        self.belief_span = init()
```
