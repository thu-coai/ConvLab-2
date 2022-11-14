# PPO

A policy optimization method in policy based reinforcement learning that uses
multiple epochs of stochastic gradient ascent and a constant
clipping mechanism as the soft constraint to perform each policy update. We adapt PPO to the dialog policy.

## Train

Run `train.py` in the `ppo` directory:

```bash
python train.py
```

For better performance, we can do immitating learning before reinforcement learning. The immitating learning is implemented in the `mle` directory.

For example, if the trained model of immitating learning is saved at FOLDER_OF_MODEL/best_mle.pol.mdl, then you can run

```bash
python train.py --load_path FOLDER_OF_MODEL/best_mle
```

Note that the *.pol.mdl* suffix should not appear in the --load_path argument.

## Trained Model

Performance:

| Task Success Rate |
| ------------ |
| 0.74 |

The model can be downloaded from: 

https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/ppo_policy_multiwoz.zip

## Reference

```
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```