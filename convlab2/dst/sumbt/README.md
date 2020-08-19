# SUMBT on Multiwoz

SUMBT (Slot-Utterance Matching Belief Tracker) is a belief tracking model that
utilizes semantic similarity between dialogue utterances and slot-values
, which is proposed by [Hwaran Lee et al., 2019](https://www.aclweb.org/anthology/P19-1546.pdf).

The code derives from [github](https://github.com/SKTBrain/SUMBT). We modify it to support user DST. 

## Usage


### Train & Evaluate

from Convlab root directory
```python
from convlab2.dst.sumbt.multiwoz.sumbt import *
m = SUMBTTracker()
m.train()  # will train and output the model checkpoint in the output_path defined in 'sumbt_config.py' file
# m.test(mode, model_path)  # where testset in ['dev', 'test'], respectively run evaluation on dev/test set of MultiWoz, model_path specify the model you want to evaluate with. will create 2 files containing evaluation metrics in the output_path defined in config file.

```


### Track
from Convlab root directory
```python
from convlab2.dst.sumbt.multiwoz.sumbt import *
test_update() 
```

At the first run, the SumbtTracker will download a pre-trained model and save it into 'downloaded_model/' directory.

## Data

We use the multiwoz data.

## Performance on Multiwoz

We evaluate the Joint accuracy and Slot accuracy on Multiwoz 2.0 validation and test set. 
The accuracy on validation set are slightly higher than the results reported in the paper,
because in the evaluation code all undefined values in ontology are set `none` but predictions 
will always be wrong for all undefined domain-slots.  

|   | Joint acc  | Slot acc    | Joint acc (Restaurant)  |  Slot acc (Restaurant)|
| ----- | ----- | ------ | ------ | ----    |
| dev     | 0.47 | 0.97 | 0.83 | 0.97  |
| test    | 0.51 | 0.97 | 0.84 | 0.97

## Model Structure

SUMBT considers a domain-slot type (e.g., 'restaurant-food') as a query and finds the corresponding 
slot-value in a pair of system-user utterances, under the assumption that the answer appear in the utterances.

The model encodes domain-slot with a fixed BERT model and encodes utterances with another BERT 
of which parameters are fine-tuned during training. A MultiHead attention layer is
employed to capture slot-specific information, and the attention context vector is fed
into an RNN to model the flow of dialogues.


## Reference

```
@inproceedings{lee2019sumbt,
  title={SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking},
  author={Lee, Hwaran and Lee, Jinsik and Kim, Tae-Yoon},
  booktitle={Proceedings of the 57th Conference of the Association for Computational Linguistics},
  pages={5478--5483},
  year={2019}
}
```

