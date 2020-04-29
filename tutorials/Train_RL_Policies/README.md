# Train RL Policies
In the task-oriented system, a common strategy of learning a reinforcement learning dialog policy offline is to build a user simulator and make simulated interactions between the policy and simulator.

## Build an Environment
In convlab2, we provide the enviroment class for training an RL policy, and we regard the all the components except system policy as the environment, i.e.:

```python
from convlab2.dialog_agent.env import Environment

# We don't need NLU, DST and NLG for user simulator
policy_usr = RulePolicy(character='usr')
simulator = PipelineAgent(None, None, policy_usr, None, 'user')

# We don't need NLU and NLG for system
dst_sys = RuleDST()

evaluator = MultiWozEvaluator()
env = Environment(None, simulator, None, dst_sys, evaluator)
```

## Collect dialog sessions with multi-processing
First, we instantiate our system policy to train:

```python
policy_sys = PPO(is_train=True)
```

To sample dialog sessions in a distributed setting, each process contains an actor that acts in its own copy of the environment.

```python
# sample data asynchronously
batch = sample(env, policy_sys, batchsz, process_num)
```

Then, it returns all the samples to the main process and update the policy.

```python
# data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
# batch.action: ([1, a_dim], [1, a_dim]...)
# batch.reward/ batch.mask: ([1], [1]...)
s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
batchsz_real = s.size(0)

policy_sys.update(epoch, batchsz_real, s, a, r, mask)
```

## Run
Please refer to example_train.py in this directory for details.

```bash
$ python example_train.py
```

You can change the following arguments in example_train.py,

```python
parser.add_argument("--batchsz", type=int, default=1024, help="batch size of trajactory sampling")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs to train")
parser.add_argument("--process_num", type=int, default=8, help="number of processes of trajactory sampling")
```
or `config.json` of corresponding RL policy during the training.