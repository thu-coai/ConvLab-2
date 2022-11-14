# Imitation on multiwoz

Vanilla MLE Policy employs a multi-class classification via Imitation Learning with a set of compositional actions where a compositional action consists of a set of dialog act items.

## Train

```
python train.py
```

You can modify *config.json* to change the setting.

## Data

data/multiwoz/[train/val/test].json

## Trained Model

Performance:

| Task Success Rate |
| ------------ |
| 0.56 |

The model can be downloaded from: 

https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/mle_policy_multiwoz.zip
