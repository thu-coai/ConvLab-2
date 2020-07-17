The human evaluation tool is based on ParlAI, and performed on Amazon Mechanic Turk. We suggest getting yourself familiar with the tools as described here:
- ParlAI Setup: https://parl.ai/#getstarted.
- Mturk Tutorial: https://parl.ai/docs/tutorial_mturk.html.

To run the human evaluation, please first run `run_agent.py` as the bot/agent service on your local machine or server. Then run `run.py` to initiate the ParlAI tool. You can change the code from line 27-32 of `run_agent.py` to set up your agent. Please refer to the [tutorial](https://github.com/thu-coai/ConvLab-2/blob/master/tutorials/Getting_Started.ipynb) for more details of building an agent. 

Note that the agent bot is designed to be stateless. The POST parameters have dependencies on agent states that describes all the information needed for an agent to generate responses. If you change the structure of a pipeline agent, e.g., an end2end model, please ensure that the agent state contains all the information for prediction.
