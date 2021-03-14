# README

## Features

- Annotations: dialogue act, character-level span for non-categorical slots. state and state updates.   

Statistics: 

|       | \# dialogues | \# utterances | avg. turns | avg. tokens | \# domains |
| ----- | ------------ | ------------- | ---------- | ----------- | ---------- |
| train | 8434         | 105066         | 12.46     | 17.27      | 7          |
| dev | 999         | 13731         | 13.74      | 17.72       | 7          |
| train | 1000         | 13744         | 13.74       | 17.67       | 7          |


## Main changes

- only keep 5 domains in state annotations and dialog acts. 
- `pricerange`, `area`, `day`, `internet`, `parking`, `stars` are considered categorical slots.
- punctuation marks are split from their previous tokens. e.g `I want to find a hotel. -> 
  I want to find a hotel .`

Run `evaluate.py`:

da values match rate:    97.944
state values match rate: 66.945

### original data

- from [multiwoz](https://github.com/budzianowski/multiwoz) repo.
- original multiwoz2.2 dataset gives slot value in List format. We take the first value 
in each slot list as ground-truth value.


