Human evaluation tool is based on ParlAI.

Please setup the tool following steps at https://parl.ai/#getstarted.

Details about how to use it can be found at https://parl.ai/docs/tutorial_mturk.html.

Build your agent in `run_agent.py` line 27-32. Please refer to the [tutorial](https://github.com/thu-coai/ConvLab-2/blob/master/tutorials/Getting_Started.ipynb) for building an agent. 

Note that the POST parameters have dependencies on agent states that describes all the information needed for an agent to generate responses. If you change the structure of a pipeline agent, e.g. an end2end model, please ensure that the agent state contains all the information for prediction.