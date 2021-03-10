# README

## Features

- Annotations: dialogue act, character-level span for non-categorical slots.

Statistics: 

|       | \# dialogues | \# utterances | avg. turns | avg. tokens | \# domains |
| ----- | ------------ | ------------- | ---------- | ----------- | ---------- |
| train | 406         | 2936         | 7.23     | 11.36       | 1          |
| dev | 135         | 941         | 6.97      | 11.99       | 1          |
| train | 135         | 935         | 6.93       | 11.87       | 1          |

## Main changes

- domain is set to **restaurant**
- ignore some rare pair
- 3 values are not found in original utterances
- **dontcare** values in non-categorical slots are calculated in `evaluate.py` so `da_match` in evaluation is lower than actual number.

## Original data

camrest used in convlab2, included in `data/` path