# VHUS on multiwoz

A data driven variational hierarchical seq2seq user simulator where an unobserved latent random variable generates the user turn sequence. 

- Hierarchical encoder to encode system response, user goal, and dialog history.
- A variational sampling step before user turn decoder is proposed to generate a slightly different user query.

We implemented VHUS on multiwoz dataset. The original paper can be found at [IEEE Xplore Digital Library](https://ieeexplore.ieee.org/abstract/document/8639652/).

## Train

```python
python train.py
```

You can modify *config.json* to change the setting.

## Data

data/multiwoz/annotated_user_da_with_span_full.json

## Trained Model

Note: we have not tuned this model elaborately.

Performance:

| Task Success Rate |
| ------------ |
| 0.31 |

The model can be downloaded from: 

https://convlab.blob.core.windows.net/convlab-2/vhus_simulator_multiwoz.zip

## Reference

```
@inproceedings{gur2018user,
  title={User modeling for task oriented dialogues},
  author={G{\"u}r, Izzeddin and Hakkani-T{\"u}r, Dilek and T{\"u}r, Gokhan and Shah, Pararth},
  booktitle={2018 IEEE Spoken Language Technology Workshop (SLT)},
  pages={900--906},
  year={2018},
  organization={IEEE}
}
```