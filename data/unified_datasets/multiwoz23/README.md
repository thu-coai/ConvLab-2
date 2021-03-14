# README

## Features

- Annotations: dialogue act, character-level span for non-categorical slots. state and state updates.   

Statistics: 

|       | \# dialogues | \# utterances | avg. turns | avg. tokens | \# domains |
| ----- | ------------ | ------------- | ---------- | ----------- | ---------- |
| train | 8434         | 105066         | 12.46     | 15.75      | 7          |
| dev | 999         | 13731         | 13.74      | 16.1       | 7          |
| train | 1000         | 13744         | 13.74       | 16.08       | 7          |


## Main changes

- only keep 5 domains in state annotations and dialog acts.
- `pricerange`, `area`, `day`, `internet`, `parking`, `stars` are considered categorical slots.
- replace special tokens by space. e.g. `I want@to find a hotel.  ->  I want to find a hotel.`

Run `evaluate.py`:

da values match rate:    98.798
state values match rate: 89.185

### original data

- from [multiwoz-coref](https://github.com/lexmen318/MultiWOZ-coref) repo.
- slot description by multiwoz2.2
- some hand-written descriptions. 


