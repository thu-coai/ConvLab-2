# README

## Features

- Annotations: character-level span for non-categorical slots. No slot descriptions.

Statistics: 

|       | \# dialogues | \# utterances | avg. turns | avg. tokens | \# domains |
| ----- | ------------ | ------------- | ---------- | ----------- | ---------- |
| train | 30483        | 540311        | 17.72      | 9.18        | 13         |

## Main changes

- each speaker for one turn
- intent is set to **inform**
- not annotate state and state upadte
- span info is provided by original data

## Original data

https://github.com/google-research-datasets/Taskmaster

TM-1: https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019

TM-2: https://github.com/google-research-datasets/Taskmaster/tree/master/TM-2-2020