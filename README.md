# ConvLab-2
[![Build Status](https://travis-ci.com/thu-coai/ConvLab-2.svg?branch=master)](https://travis-ci.com/thu-coai/ConvLab-2)

**ConvLab-2** is an open-source toolkit that enables researchers to build task-oriented dialogue systems with state-of-the-art models, perform an end-to-end evaluation, and diagnose the weakness of systems. As the successor of [ConvLab](https://github.com/ConvLab/ConvLab), ConvLab-2 inherits ConvLab's framework but integrates more powerful dialogue models and supports more datasets. Besides, we have developed an analysis tool and an interactive tool to assist researchers in diagnosing dialogue systems. [[paper]](https://arxiv.org/abs/2002.04793)

- [Installation](#installation)
- [Tutorials](#tutorials)
- [Documents](#documents)
- [Models](#models)
- [Supported Datasets](#Supported-Datasets)
- [End-to-end Performance on MultiWOZ](#End-to-end-Performance-on-MultiWOZ)
- [Module Performance on MultiWOZ](#Module-Performance-on-MultiWOZ)
- [Issues](#issues)
- [Contributions](#contributions)
- [Citing](#citing)
- [License](#license)

## Installation

Require python 3.6.

Clone this repository:
```bash
git clone https://github.com/thu-coai/ConvLab-2.git
```

Install ConvLab-2 via pip:

```bash
cd ConvLab-2
pip install -e .
```

## Tutorials

- [Getting Started](https://github.com/thu-coai/ConvLab-2/blob/master/tutorials/Getting_Started.ipynb) (Have a try on [Colab](https://colab.research.google.com/github/thu-coai/ConvLab-2/blob/master/tutorials/Getting_Started.ipynb)!)
- [Add New Model](https://github.com/thu-coai/ConvLab-2/blob/master/tutorials/Add_New_Model.md)
- [Train RL Policies](https://github.com/thu-coai/ConvLab-2/blob/master/tutorials/Train_RL_Policies)
- [Interactive Tool](https://github.com/thu-coai/ConvLab-2/blob/master/deploy) [[demo video]](https://youtu.be/00VWzbcx26E)

## Documents
Our documents are on https://thu-coai.github.io/ConvLab-2_docs/convlab2.html.

## Models

We provide following models:

- NLU: SVMNLU, MILU, BERTNLU
- DST: rule, MDBT, TRADE, SUMBT
- Policy: rule, Imitation, REINFORCE, PPO, GDPL, MDRG, HDSA, LaRL
- Simulator policy: Agenda, VHUS
- NLG: Template, SCLSTM
- End2End: Sequicity, DAMD, RNN_rollout

For  more details about these models, You can refer to `README.md` under `convlab2/$module/$model/$dataset` dir such as `convlab2/nlu/jointBERT/multiwoz/README.md`.

## Supported Datasets

- [Multiwoz 2.1](https://github.com/budzianowski/multiwoz)
  - We add user dialogue act (*inform*, *request*, *bye*, *greet*, *thank*), remove 5 sessions that have incomplete dialogue act annotation and place it under `data/multiwoz` dir.
  - Train/val/test size: 8434/999/1000. Split as original data.
  - LICENSE: Attribution 4.0 International, url: http://creativecommons.org/licenses/by/4.0/
- [CrossWOZ](https://github.com/thu-coai/CrossWOZ)
  - We offers a rule-based user simulator and a complete set of models for building a pipeline system on the CrossWOZ dataset. We correct few state annotation and place it under `data/crosswoz` dir.
  - Train/val/test size: 5012/500/500. Split as original data.
  - LICENSE: Attribution 4.0 International, url: http://creativecommons.org/licenses/by/4.0/
- [Camrest](https://www.repository.cam.ac.uk/handle/1810/260970)
  - We add system dialogue act (*inform*, *request*, *nooffer*) and place it under `data/camrest` dir.
  - Train/val/test size: 406/135/135. Split as original data.
  - LICENSE: Attribution 4.0 International, url: http://creativecommons.org/licenses/by/4.0/
- [Dealornot](https://github.com/facebookresearch/end-to-end-negotiator/tree/master/src/data/negotiate)
  - Placed under `data/dealornot` dir.
  - Train/val/test size: 5048/234/526. Split as original data.
  - LICENSE: Attribution-NonCommercial 4.0 International, url: https://creativecommons.org/licenses/by-nc/4.0/

## End-to-end Performance on MultiWOZ

*Notice*: The results are for commits before [`bdc9dba`](https://github.com/thu-coai/ConvLab-2/commit/bdc9dba72c957d97788e533f9458ed03a4b0137b) (inclusive). We will update the results after improving user policy.

We perform end-to-end evaluation (1000 dialogues) on MultiWOZ using the user simulator below (a full example on `tests/test_end2end.py`) :

```python
# BERT nlu trained on sys utterance
user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json', model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
user_dst = None
user_policy = RulePolicy(character='usr')
user_nlg = TemplateNLG(is_user=True)
user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

set_seed(20200202)
analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='sys_agent', total_dialog=1000)
```

Main metrics (refer to `convlab2/evaluator/multiwoz_eval.py` for more details):

- Complete: whether complete the goal. Judged by the Agenda policy instead of external evaluator.
- Success: whether all user requests have been informed and the booked entities satisfy the constraints.
- Book: how many the booked entities satisfy the user constraints.
- Inform Precision/Recall/F1: how many user requests have been informed.
- Turn(succ/all): average turn number for successful/all dialogues.

Performance (the first row is the default config for each module. Empty entries are set to default config.):

| NLU         | DST       | Policy         | NLG         | Complete rate | Success rate | Book rate | Inform P/R/F1 | Turn(succ/all) |
| ----------- | --------- | -------------- | ----------- | ------------- | ------------ | --------- | --------- | -------------- |
| **BERTNLU** | RuleDST   | RulePolicy     | TemplateNLG |   90.5       |     81.3    |   91.1 | 79.7/92.6/83.5 | 11.6/12.3      |
| **MILU**    | RuleDST | RulePolicy | TemplateNLG |    93.3       |   81.8      |   93.0    | 80.4/94.7/84.8 | 11.3/12.1      |
| BERTNLU | RuleDST | RulePolicy | **SCLSTM**  |   48.5    | 40.2 | 56.9   | 62.3/62.5/58.7 |  11.9/27.1         |
| BERTNLU     | RuleDST | **MLEPolicy**  | TemplateNLG |     42.7          |    35.9      |  17.6   | 62.8/69.8/62.9  |  12.1/24.1    |
| BERTNLU | RuleDST | **PGPolicy**   | TemplateNLG |     37.4         |    31.7     |   17.4  |  57.4/63.7/56.9  |   11.0/25.3    |
| BERTNLU | RuleDST | **PPOPolicy**  | TemplateNLG |     61.1         |    44.0    |   44.6    | 63.9/76.8/67.2  |  12.5/20.8   |
| BERTNLU | RuleDST | **GDPLPolicy** | TemplateNLG |     49.4         |     38.4    |  20.1     |  64.5/73.8/65.6 |  11.5/21.3    |
| None        | **TRADE** | RulePolicy | TemplateNLG |    32.4      |    20.1     |    34.7      |  46.9/48.5/44.0 |  11.4/23.9      |
| None        | **SUMBT** | RulePolicy | TemplateNLG |   34.5       |   29.4     |   62.4    |  54.1/50.3/48.3  |   11.0/28.1     |
| BERTNLU | RuleDST | **MDRG**       | None        | 21.6 | 17.8 | 31.2 | 39.9/36.3/34.8 | 15.6/30.5|
| BERTNLU | RuleDST | **LaRL**       | None        | 34.8 | 27.0 | 29.6 | 49.1/53.6/47.8 |13.2/24.4|
| None | **SUMBT** | **LaRL** | None |  32.9 | 23.7  |  25.9 | 48.6/52.0/46.7 | 12.5/24.3|
| None | None | **DAMD***      | None | 39.5| 34.3 | 51.4 | 60.4/59.8/56.3 | 15.8/29.8 |

*: end-to-end models used as sys_agent directly.

## Module Performance on MultiWOZ

### NLU

By running `convlab2/nlu/evaluate.py MultiWOZ $model all`:

|         | Precision | Recall | F1    |
| ------- | --------- | ------ | ----- |
| BERTNLU | 82.48     | 85.59  | 84.01 |
| MILU    | 80.29     | 83.63  | 81.92 |
| SVMNLU  | 74.96     | 50.74  | 60.52 |

### DST 

By running `convlab2/dst/evaluate.py MultiWOZ $model`:

|             |  Joint accuracy  | Slot accuracy | Joint F1  |
| --------    |   -------------   | -------------  | --------|
|  MDBT       |   0.06           |      0.89       | 0.43    |
|  SUMBT      |    0.30         |       0.96       | 0.83    |
|   TRADE     |    0.40         |       0.96       | 0.84    |

### Policy

*Notice*: The results are for commits before [`bdc9dba`](https://github.com/thu-coai/ConvLab-2/commit/bdc9dba72c957d97788e533f9458ed03a4b0137b) (inclusive). We will update the results after improving user policy.

By running `convlab2/policy/evalutate.py --model_name $model`

|           | Task Success Rate |
| --------- | ----------------- |
| MLE       | 0.56              |
| PG        | 0.54              |
| PPO       | 0.74              |
| GDPL      | 0.58              |

### NLG

By running `convlab2/nlg/evaluate.py MultiWOZ $model sys`

|          | corpus BLEU-4 |
| -------- | ------------- |
| Template | 0.3309        |
| SCLSTM   | 0.4884        |

## Translation-train SUMBT for cross-lingual DST

### Train

With Convlab-2, you can train SUMBT on a machine-translated dataset like this:

```python
# train.py
import os
from sys import argv

if __name__ == "__main__":
    if len(argv) != 2:
        print('usage: python3 train.py [dataset]')
        exit(1)
    assert argv[1] in ['multiwoz', 'crosswoz']

    from convlab2.dst.sumbt.multiwoz_zh.sumbt import SUMBT_PATH
    if argv[1] == 'multiwoz':
        from convlab2.dst.sumbt.multiwoz_zh.sumbt import SUMBTTracker as SUMBT
    elif argv[1] == 'crosswoz':
        from convlab2.dst.sumbt.crosswoz_en.sumbt import SUMBTTracker as SUMBT

    sumbt = SUMBT()
    sumbt.train(True)
```

### Evaluate

Execute `evaluate.py` (under `convlab2/dst/`) with following command:

```bash
python3 evaluate.py [CrossWOZ-en|MultiWOZ-zh] [val|test|human_val]
```

evaluation of our pre-trained models are: (joint acc.)

| type  | CrossWOZ-en | MultiWOZ-zh |
| ----- | ----------- | ----------- |
| val   | 12.4%       | 45.1%       |
| test  | 12.4%       | 43.5%       |
| human_val | 10.6%       | 49.4%       |

`human_val` option will make the model evaluate on the validation set translated by human. 

Note: You may want to download pre-traiend BERT models and translation-train SUMBT models provided by us.

Without modifying any code, you could:

- download pre-trained BERT models from:

  - [bert-base-uncased](https://huggingface.co/bert-base-uncased)  for CrossWOZ-en
  - [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)  for MultiWOZ-zh

  extract it to `./pre-trained-models`.

- for translation-train SUMBT model:

  - [trained on CrossWOZ-en](https://convlab.blob.core.windows.net/convlab-2/crosswoz_en-pytorch_model.bin.zip)
  - [trained on MultiWOZ-zh](https://convlab.blob.core.windows.net/convlab-2/multiwoz_zh-pytorch_model.bin.zip)
  - Say the data set is CrossWOZ (English), (after extraction) just save the pre-trained model under `./convlab2/dst/sumbt/crosswoz_en/pre-trained` and name it with `pytorch_model.bin`. 

## Issues

You are welcome to create an issue if you want to request a feature, report a bug or ask a general question.

## Contributions

We welcome contributions from community.

- If you want to make a big change, we recommend first creating an issue with your design.
- Small contributions can be directly made by a pull request.
- If you like make contributions to our library, see issues to find what we need.

## Team

**ConvLab-2** is maintained and developed by Tsinghua University Conversational AI group (THU-coai) and Microsoft Research (MSR).

We would like to thank:

Yan Fang, Zhuoer Feng, Jianfeng Gao, Qihan Guo, Kaili Huang, Minlie Huang, Sungjin Lee, Bing Li, Jinchao Li, Xiang Li, Xiujun Li, Lingxiao Luo, Wenchang Ma, Mehrad Moradshahi, Baolin Peng, Runze Liang, Ryuichi Takanobu, Hongru Wang, Jiaxin Wen, Yaoqin Zhang, Zheng Zhang, Qi Zhu, Xiaoyan Zhu.


## Citing

If you use ConvLab-2 in your research, please cite:

```
@inproceedings{zhu2020convlab2,
    title={ConvLab-2: An Open-Source Toolkit for Building, Evaluating, and Diagnosing Dialogue Systems},
    author={Qi Zhu and Zheng Zhang and Yan Fang and Xiang Li and Ryuichi Takanobu and Jinchao Li and Baolin Peng and Jianfeng Gao and Xiaoyan Zhu and Minlie Huang},
    year={2020},
    booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
}
```

## License

Apache License 2.0
