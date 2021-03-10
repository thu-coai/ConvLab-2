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
- make some rule-based fixes on categorical values to make them in `possible value` lists
- `belief_states` in WOZ dataset contains `request` intents, which are ignored in processing
- some state annotations are not consistent with dialogue_act annotations. for example in `woz_train_en.json`, first dialog, 2nd turn:
 
    `user: "How about Chinese food?"`
   
    `chinese food` is included in `dialogue_act` annotation as a `inform` intent, but not updated in `belief_state` annotation.
    
    

## Original data

https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz