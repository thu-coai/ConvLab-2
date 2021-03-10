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

- manually clean characters starting with `\u`
- only hotel-parking and hotel-internet are considered categorical slots.
- `dontcare` has no span annotations.
- only keep 5 domains in state annotations and dialog acts. 
- `booking` domain appears in dialog acts but not in states.
- some dialogs contain categorical slot change from `value` to empty value, and some of them are correct while some wrong. 


### original data

- from convlab repo.
- slot description by multiwoz2.2
- some hand-written descriptions. 


