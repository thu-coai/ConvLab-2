# GDPL

A join policy optimization and reward estimation method using adversarial inverse reinforcement learning that learns a dialog policy and builds a reward estimator simultaneously. The reward estimator evaluates the state-action pairs to guide the dialog policy at each dialog turn.

## Train

Run `train.py` in the `gdpl` directory:

```bash
python train.py
```

For better performance, we can do immitating learning before reinforcement learning. The immitating learning is implemented in the `mle` directory.

For example, if the trained model of immitating learning is saved at FOLDER_OF_MODEL/best_mle.pol.mdl, then you can run

```bash
python train.py --load_path FOLDER_OF_MODEL/best_mle
```

Note that the *.pol.mdl* suffix should not appear in the --load_path argument.

## Reference

```
@inproceedings{takanobu2019guided,
  title={Guided Dialog Policy Learning: Reward Estimation for Multi-Domain Task-Oriented Dialog},
  author={Takanobu, Ryuichi and Zhu, Hanlin and Huang, Minlie},
  booktitle={EMNLP-IJCNLP},
  pages={100--110},
  year={2019}
}
```