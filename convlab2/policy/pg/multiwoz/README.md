# REINFORCE

A simple stochastic gradient algorithm for policy gradient reinforcement learning. We adapt REINFORCE to the dialog policy.

## Train

Run `train.py` in the `pg` directory:

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
| 0.54 |

The model can be downloaded from: 

https://convlab.blob.core.windows.net/convlab-2/pg_policy_multiwoz.zip

## Reference

```
@article{williams1992simple,
  title={Simple statistical gradient-following algorithms for connectionist reinforcement learning},
  author={Williams, Ronald J},
  journal={Machine learning},
  volume={8},
  number={3-4},
  pages={229--256},
  year={1992},
  publisher={Springer}
}
```