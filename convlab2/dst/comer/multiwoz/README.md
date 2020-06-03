## COMER: Scalable and Accurate Dialogue State Tracking via Hierarchical Sequence Generation

This is the PyTorch implementation of the paper:
**Sclable and Accurate Dialogue State Tracking via Hierachical Sequence Generation**. Liliang Ren,Jianmo Ni, Julian McAuley. ***EMNLP 2019***
[[PDF]](https://arxiv.org/abs/1909.00754) [[Slides]](https://drive.google.com/file/d/1p3SQN_xPNsUGqS4x1XHJ0ySigbG1dcsL/view?usp=sharing) [[Poster]](https://drive.google.com/file/d/1cqiW7QtX_EvgzIlh9DhZnNTwBwPoQ4LS/view?usp=sharing) [[Trained Model]](https://drive.google.com/a/eng.ucsd.edu/file/d/1KiL9yoAhKBuGXLQA3IF5TvgnFdSDecT-/view?usp=drivesdk)

The code is written and tested with PyTorch == 1.1.0. The minimum requirement of the graphic memory is 8GB for training on the MultiWoZ dataset.

**\*\*\*\*\* New October 20th, 2019: Updated Empirical Results \*\*\*\*\***

We did some modifications to the regularization and the evaluation of our model and achieved a joint goal accuracy of 48.79% on the MultiWoZ dataset. For more details, please refer to the Table 5 in the paper linked above.

## Abstract
Existing approaches to dialogue state tracking rely on pre-defined ontologies consisting of a set of all possible slot types and values. Though such approaches exhibit promising performance on single-domain benchmarks, they suffer from computational complexity that increases proportionally to the number of pre-defined slots that need tracking. This issue becomes more severe when it comes to multi-domain dialogues which include larger numbers of slots. In this paper, we investigate how to approach DST using a generation framework without the pre-defined ontology list. Given each turn of user utterance and system response, we directly generate a sequence of belief states by applying a hierarchical encoder-decoder structure. In this way, the computational complexity of our model will be a constant regardless of the number of pre-defined slots. Experiments on both the multi-domain and the single domain dialogue state tracking dataset show that our model not only scales easily with the increasing number of pre-defined domains and slots but also reaches the state-of-the-art performance.


## Create Data
```
python3 create_data.py 
```
***************************************************************


## Preprocessing
```
python3 convert_mw.py
python3 preprocess_mw.py 
python3 make_emb.py
```

***************************************************************

## Training
```
bash run.sh
```

****************************************************************

## Evaluation
```
python3 predict.py 
```

*******************************************************************
## Citation
```
@inproceedings{ren-etal-2019-scalable,
    title = "Scalable and Accurate Dialogue State Tracking via Hierarchical Sequence Generation",
    author = "Ren, Liliang  and
      Ni, Jianmo  and
      McAuley, Julian",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1196",
    doi = "10.18653/v1/D19-1196",
    pages = "1876--1885",
}
```
