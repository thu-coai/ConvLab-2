# MultiWOZ 2.3

MultiWOZ 2.3 adds co-reference annotations in addition to corrections of dialogue acts and dialogue states. Please refer to the following paper to get more details:
**MultiWOZ 2.3: A multi-domain task-oriented dialogue dataset enhanced with annotation corrections and co-reference annotation**. [[PDF]](https://arxiv.org/abs/2010.05594) (Jun. 14, 2021 updated)

Appendices [[PDF]](https://github.com/lexmen318/MultiWOZ-coref/blob/main/appendix_new.pdf) for MultiWOZ 2.3 accepted by NLPCC 2021

If you find our dataset useful and use in your work, please cite the following paper. The bibtex is listed below
<pre>
@article{han2020multiwoz,
  title={MultiWOZ 2.3: A multi-domain task-oriented dialogue dataset enhanced with annotation corrections and co-reference annotation},
  author={Han, Ting and Liu, Ximing and Takanobu, Ryuichi and Lian, Yixin and Huang, Chongxuan and Wan, Dazhen and Peng, Wei and Huang, Minlie},
  journal={arXiv preprint arXiv:2010.05594},
  year={2020}
}
</pre>

## Dataset

Three files are included in the zip file:

1. **data.json**: the updated dataset, we add co-reference annotations.
2. **dialogue_acts.json**: the updated dialogue acts. 
3. **ontology.json**: the ontology is based on MultiWOZ 2.1 and the only difference is slot format, from domain-semi-slot to domain-slot.

All files have similar format as those of previous datasets (https://github.com/budzianowski/multiwoz).

Except for the corrected and co-reference annotations, we also made the following improvements:

1. The field of "turn_id" is added to all utterances so that they could be referred in co-reference annotations.
2. There are five dialogues having no "dialogue_act" annotations in MultiWOZ 2.1. These dialogues are annotated manually one by one in MultiWOZ-coref 
3. Fixed some garbage characters inside MultiWOZ 2.1
4. The field of "new_goal" is added to all dialogues. The new goal annotations are extracted from the goal descriptions. Note that, "book" and "fail_book" ("info" and "fail_info") are merged, and redundant domains are removed.

```json
// goal
{
    "restaurant": {
        "fail_book": {
            "time": "18:30"
        },
        "book": {
            "time": "17:30",
            "people": "8"
        }
    },
    "train": {}
}
```

```json
// new_goal
{
    "restaurant": {
        "book": {
            "time": ["18:30", "17:30"],
            "people": ["8"]
        }
    }
}
```



## Experiment

The two models, SUMBT and TRADE, used in the experiment of the paper can be accessed through following links:

SUMBT: https://github.com/SKTBrain/SUMBT <br/>
TRADE: https://github.com/jasonwu0731/trade-dst <br/>

Please use the scripts provided by the two models to format the data appropriately before you run the models. The ontology comes with MultiWOZ 2.3 is based on the version in MultiWOZ 2.1 and can be directly used for the above two models. The only difference is the format of slot names. Please note that you can freely build up your own ontology.

## More experiments
On availability, we test our dataset on different DST models. Process scripts for different DST models remain unchanged and are available from their githubs (click the model name if the githubs are still accessible).

| DST Model | MultiWOZ 2.1 | MultiWOZ 2.2 | MultiWOZ 2.3 |
| --------- | ------------ | ------------ | ------------ |
| [TRADE](https://github.com/jasonwu0731/trade-dst) | 46.0% | 45.4% | 49.2% |
| [SUMBT](https://github.com/SKTBrain/SUMBT) | 49.2% | 49.7% | 52.9% |
| [COMER](https://github.com/renll/ComerNet) | 48.8% | -- | 50.2% |
| [DSTQA](https://github.com/alexa/dstqa) | 51.2% | --| 51.8% |
| [SOM-DST](https://github.com/clovaai/som-dst) | 53.1% | -- | 55.5% |
| [TripPy](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public) | 55.3% |  -- | 63.0% |
| [SimpleTOD*](https://github.com/salesforce/simpletod) | 50.3% (55.7%) | -- | 51.3% |
| [ConvBERT-DG-Multi](https://github.com/alexa/dialoglue) | 58.7% | -- | 67.9% |
| [SAVN](https://github.com/wyxlzsq/savn) | 54.5% | -- | 58.0% |

Please note that "--" means that no performence reported. * in SimpleTOD means that we only run the code for DST by keeping `dontcare` and `none`. For further details, please refer to the github: https://github.com/salesforce/simpletod.
