# README

## Features

- Annotations: dialogue act, character-level span for non-categorical slots.

Statistics: 

|       | \# dialogues | \# utterances | avg. turns | avg. tokens | \# domains |
| ----- | ------------ | ------------- | ---------- | ----------- | ---------- |
| train | 1369         | 19445         | 14.2       | 12.71       | 1          |

## Main changes

- domain is set to **travel**
- slot-value pair changes: intent-book => book-"True", action-book => booked-"True"
- ignore some rare pair
- not annotate state and state upadte
- span info is from string matching, covering 96.4 non-categorical value

## Original data

https://www.microsoft.com/en-us/research/project/frames-dataset/#!download