# Unified data format with example

Under `data/unified_datasets` directory.

single turn->dialogue with one turn

Each dataset have at least 4 files:

- `README.md`: dataset description and the main changes from original data to processed data.
- `preprocess.py`: python script that preprocess the data. By running `python preprocess.py` we can get the following two files. The structure `preprocess.py`  should be:

```python
def preprocess():
    pass
if __name__ == '__main__':
    preprocess()
```

- `ontology.json`: dataset ontology, contains descriptions, state definition, etc.
- `data.json.zip`: contains `data.json`.

### README

- Data source: publication, original data download link, etc.
- Data description:
  - Annotations: whether have dialogue act, belief state annotation.
  - Statistics: \# domains, # dialogues, \# utterances, Avg. turns, Avg. tokens (split by space), etc.
- Main changes from original data to processed data.

### Ontology

`ontology.json`: a *dict* containing:

- `domains`: (*dict*) descriptions for domains, slots. Must contains all slots in the state and non-binary dialogue acts.
  - `$domain_name`: (*dict*)
    - `description`: (*str*) description for this domain.
    - `slots`: (*dict*)
      - `$slot_name`: (*dict*)
        - `description`: (*str*) description for this slot.
        - `is_categorical`: (*bool*) categorical slot or not.
        - `possible_values`: (*list*) List of possible values the slot can take. If the slot is a categorical slot, it is a complete list of all the possible values. If the slot is a non categorical slot, it is either an empty list or a small sample of all the values taken by the slot.

- `intents`: (*dict*) descriptions for intents.
  - `$intent_name`: (*dict*)
    - `description`: (*str*) description for this intent.
- `binary_dialogue_act`: (*list* of *dict*) special dialogue acts that the value may not present in the utterance, e.g. request the address of a hotel.
  - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. domain, slot, value may be empty.
- `state`: (*dict*) belief state of all domains.
  - `$domain_name`: (*dict*)
    - `$slot_name: ""`: slot with empty value. Note that the slot set are the subset of the slot set in Part 1 definition.

### Dialogues

`data.json`: a *list* of dialogues containing:

- `dataset`: (*str*) dataset name, must be one of  ['schema', 'multiwoz', 'camrest', 'woz', ...], and be the same as the current dataset.
- `data_split`: (*str*) in [train, val, test].
- `dialogue_id`: (*str*) use dataset name as prefix, add count.
- `domains`: (*list*) domains in this dialogue.
- `turns`: (*list* of *dict*)
  - `speaker`: (*str*) "user" or "system". **User side first, user side final**, "user" and "system" appear alternately?
  - `utterance`: (*str*) sentence.
  - `utt_idx`: (*int*) `turns['utt_idx']` gives current turn.
  - `dialogue_act`: (*dict*)
    - `categorical`: (*list* of *dict*) for categorical slots.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. Value sets are defined in the ontology.
    - `non-categorical` (*list* of *dict*) for non-categorical slots.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str), "start": (int), "end": (int)}`. `start` and `end` are character indexes for the value span.
    - `binary` (*list* of *dict*) for binary dialogue acts in ontology.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. Possible dialogue acts are listed in the `ontology['binary_dialogue_act']`.
  - `state`: (*dict*, optional, user side) full state are shown in `ontology['state']`.
    - `$domain_name`: (*dict*) contains all slots in this domain.
      - `$slot_name`: (*str*) value for this slot.
  - `state_update`: (*dict*, optional, user side) records the difference of states between the current turn and the last turn.
    - `categorical`: (*list* of *dict*) for categorical slots.
      - `{"domain": (str), "slot": (str), "value": (str)}`. Value sets are defined in the ontology (**dontcare** may not be included).
    - `non-categorical` (*list* of *dict*) for non-categorical slots.
      - `{"domain": (str), "slot": (str), "value": (str), "utt_idx": (int), "start": (int), "end": (int)}`. `utt_idx` is the utterance index of the value. `start` and `end` are character indexes for the value span in the current turn. `turn[utt_idx]['utterance'][start:end]` gives the value.

Other attributes are optional.

Run `python evaluate.py $dataset` to check the validation of processed dataset.

## Example of Schema Dataset

```json
	{
    "dataset": "schema",
    "data_split": "train",
    "dialogue_id": "schema_535",
    "original_id": "5_00022",
    "domains": [
      "event_2"
    ],
    "turns": [
      {
        "speaker": "user",
        "utterance": "I feel like going out to do something in Oakland. I've heard the Raiders Vs Bengals game should be good.",
        "utt_idx": 0,
        "dialogue_act": {
          "binary": [
            {
              "intent": "inform_intent",
              "domain": "event_2",
              "slot": "intent",
              "value": "geteventdates"
            }
          ],
          "categorical": [],
          "non-categorical": [
            {
              "intent": "inform",
              "domain": "event_2",
              "slot": "event_name",
              "value": "raiders vs bengals",
              "start": 65,
              "end": 83
            },
            {
              "intent": "inform",
              "domain": "event_2",
              "slot": "city",
              "value": "oakland",
              "start": 41,
              "end": 48
            }
          ]
        },
        "state": {
          "event_2": {
            "event_type": "",
            "category": "",
            "event_name": "raiders vs bengals",
            "date": "",
            "time": "",
            "number_of_tickets": "",
            "city": "oakland",
            "venue": "",
            "venue_address": ""
          }
        },
        "state_update": {
          "categorical": [],
          "non-categorical": [
            {
              "domain": "event_2",
              "slot": "city",
              "value": "oakland",
              "utt_idx": 0,
              "start": 41,
              "end": 48
            },
            {
              "domain": "event_2",
              "slot": "event_name",
              "value": "raiders vs bengals",
              "utt_idx": 0,
              "start": 65,
              "end": 83
            }
          ]
        }
      },
      {
        "speaker": "system",
        "utterance": "The Raiders Vs Bengals game is at Oakland-Alameda County Coliseum today.",
        "utt_idx": 1,
        "dialogue_act": {
          "binary": [],
          "categorical": [],
          "non-categorical": [
            {
              "intent": "offer",
              "domain": "event_2",
              "slot": "date",
              "value": "today",
              "start": 66,
              "end": 71
            },
            {
              "intent": "offer",
              "domain": "event_2",
              "slot": "event_name",
              "value": "raiders vs bengals",
              "start": 4,
              "end": 22
            },
            {
              "intent": "offer",
              "domain": "event_2",
              "slot": "venue",
              "value": "oakland-alameda county coliseum",
              "start": 34,
              "end": 65
            }
          ]
        }
      },
      {
        "speaker": "user",
        "utterance": "What time does it start?",
        "utt_idx": 2,
        "dialogue_act": {
          "binary": [
            {
              "intent": "request",
              "domain": "event_2",
              "slot": "time",
              "value": ""
            }
          ],
          "categorical": [],
          "non-categorical": []
        },
        "state": {
          "event_2": {
            "event_type": "",
            "category": "",
            "event_name": "raiders vs bengals",
            "date": "",
            "time": "",
            "number_of_tickets": "",
            "city": "oakland",
            "venue": "",
            "venue_address": ""
          }
        },
        "state_update": {
          "categorical": [],
          "non-categorical": []
        }
      },
      {
        "speaker": "system",
        "utterance": "It starts at 7 pm.",
        "utt_idx": 3,
        "dialogue_act": {
          "binary": [],
          "categorical": [],
          "non-categorical": [
            {
              "intent": "inform",
              "domain": "event_2",
              "slot": "time",
              "value": "7 pm",
              "start": 13,
              "end": 17
            }
          ]
        }
      },
      {
        "speaker": "user",
        "utterance": "That sounds fine.",
        "utt_idx": 4,
        "dialogue_act": {
          "binary": [
            {
              "intent": "select",
              "domain": "event_2",
              "slot": "",
              "value": ""
            }
          ],
          "categorical": [],
          "non-categorical": []
        },
        "state": {
          "event_2": {
            "event_type": "",
            "category": "",
            "event_name": "raiders vs bengals",
            "date": "today",
            "time": "",
            "number_of_tickets": "",
            "city": "oakland",
            "venue": "",
            "venue_address": ""
          }
        },
        "state_update": {
          "categorical": [],
          "non-categorical": [
            {
              "domain": "event_2",
              "slot": "date",
              "value": "today",
              "utt_idx": 1,
              "start": 66,
              "end": 71
            }
          ]
        }
      },
      {
        "speaker": "system",
        "utterance": "Do you want to get tickets for it?",
        "utt_idx": 5,
        "dialogue_act": {
          "binary": [
            {
              "intent": "offer_intent",
              "domain": "event_2",
              "slot": "intent",
              "value": "buyeventtickets"
            }
          ],
          "categorical": [],
          "non-categorical": []
        }
      },
      {
        "speaker": "user",
        "utterance": "Yes, can you buy 3 tickets for me?",
        "utt_idx": 6,
        "dialogue_act": {
          "binary": [
            {
              "intent": "affirm_intent",
              "domain": "event_2",
              "slot": "",
              "value": ""
            }
          ],
          "categorical": [
            {
              "intent": "inform",
              "domain": "event_2",
              "slot": "number_of_tickets",
              "value": "3"
            }
          ],
          "non-categorical": []
        },
        "state": {
          "event_2": {
            "event_type": "",
            "category": "",
            "event_name": "raiders vs bengals",
            "date": "today",
            "time": "",
            "number_of_tickets": "3",
            "city": "oakland",
            "venue": "",
            "venue_address": ""
          }
        },
        "state_update": {
          "categorical": [
            {
              "domain": "event_2",
              "slot": "number_of_tickets",
              "value": "3"
            }
          ],
          "non-categorical": []
        }
      },
      {
        "speaker": "system",
        "utterance": "Sure. I will go ahead and buy 3 tickets for the Raiders Vs Bengals game in Oakland today. Is that right?",
        "utt_idx": 7,
        "dialogue_act": {
          "binary": [],
          "categorical": [
            {
              "intent": "confirm",
              "domain": "event_2",
              "slot": "number_of_tickets",
              "value": "3"
            }
          ],
          "non-categorical": [
            {
              "intent": "confirm",
              "domain": "event_2",
              "slot": "event_name",
              "value": "raiders vs bengals",
              "start": 48,
              "end": 66
            },
            {
              "intent": "confirm",
              "domain": "event_2",
              "slot": "date",
              "value": "today",
              "start": 83,
              "end": 88
            },
            {
              "intent": "confirm",
              "domain": "event_2",
              "slot": "city",
              "value": "oakland",
              "start": 75,
              "end": 82
            }
          ]
        }
      },
      {
        "speaker": "user",
        "utterance": "Yes, that's good. What's the address?",
        "utt_idx": 8,
        "dialogue_act": {
          "binary": [
            {
              "intent": "request",
              "domain": "event_2",
              "slot": "venue_address",
              "value": ""
            },
            {
              "intent": "affirm",
              "domain": "",
              "slot": "",
              "value": ""
            }
          ],
          "categorical": [],
          "non-categorical": []
        },
        "state": {
          "event_2": {
            "event_type": "",
            "category": "",
            "event_name": "raiders vs bengals",
            "date": "today",
            "time": "",
            "number_of_tickets": "3",
            "city": "oakland",
            "venue": "",
            "venue_address": ""
          }
        },
        "state_update": {
          "categorical": [],
          "non-categorical": []
        }
      },
      {
        "speaker": "system",
        "utterance": "The game is at 7000 Coliseum Way. I've bought the tickets.",
        "utt_idx": 9,
        "dialogue_act": {
          "binary": [
            {
              "intent": "notify_success",
              "domain": "event_2",
              "slot": "",
              "value": ""
            }
          ],
          "categorical": [],
          "non-categorical": [
            {
              "intent": "inform",
              "domain": "event_2",
              "slot": "venue_address",
              "value": "7000 coliseum way",
              "start": 15,
              "end": 32
            }
          ]
        }
      },
      {
        "speaker": "user",
        "utterance": "Thanks! That's all.",
        "utt_idx": 10,
        "dialogue_act": {
          "binary": [
            {
              "intent": "thank_you",
              "domain": "",
              "slot": "",
              "value": ""
            }
          ],
          "categorical": [],
          "non-categorical": []
        },
        "state": {
          "event_2": {
            "event_type": "",
            "category": "",
            "event_name": "raiders vs bengals",
            "date": "today",
            "time": "",
            "number_of_tickets": "3",
            "city": "oakland",
            "venue": "",
            "venue_address": ""
          }
        },
        "state_update": {
          "categorical": [],
          "non-categorical": []
        }
      },
      {
        "speaker": "system",
        "utterance": "Need help with anything else?",
        "utt_idx": 11,
        "dialogue_act": {
          "binary": [
            {
              "intent": "req_more",
              "domain": "",
              "slot": "",
              "value": ""
            }
          ],
          "categorical": [],
          "non-categorical": []
        }
      },
      {
        "speaker": "user",
        "utterance": "No, thank you.",
        "utt_idx": 12,
        "dialogue_act": {
          "binary": [
            {
              "intent": "negate",
              "domain": "",
              "slot": "",
              "value": ""
            },
            {
              "intent": "thank_you",
              "domain": "",
              "slot": "",
              "value": ""
            }
          ],
          "categorical": [],
          "non-categorical": []
        },
        "state": {
          "event_2": {
            "event_type": "",
            "category": "",
            "event_name": "raiders vs bengals",
            "date": "today",
            "time": "",
            "number_of_tickets": "3",
            "city": "oakland",
            "venue": "",
            "venue_address": ""
          }
        },
        "state_update": {
          "categorical": [],
          "non-categorical": []
        }
      }
    ]
  }
```

