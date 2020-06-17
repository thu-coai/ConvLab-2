# MultiWOZ ZH

Kaili Huang

This directory contains translated MultiWOZ 2.1 dataset (from English to Chinese)

#### Ontology translation

- **vocab_dict.json**: translation of the ontology that appear in the data.
- **ontology_translate.py**: translation function that can be used to translate original value using `vocab_dict.json`.

annotation translation are categorized by slot:

| domain     | human translation                                        | machine translation                                          | template translation | don't translate            |
| ---------- | -------------------------------------------------------- | ------------------------------------------------------------ | -------------------- | -------------------------- |
| attraction | address, area, name, pricerange, type                    | choice, entrance fee, none, phone, postcode                  |                      |                            |
| booking    | name                                                     | day, none, people, stay, time                                |                      | Ref                        |
| bus        | departure, destination                                   | arriveBy, day, leaveAt, people                               |                      |                            |
| hospital   | address, department                                      | none, time                                                   |                      | phone, postcode, reference |
| hotel      | address, area, internet, name, parking, pricerange, type | Day, Stay, choice, day, none, people, phone, reference, stars,  stay |                      | People, Ref, postcode      |
| police     | Name, address                                            | none                                                         |                      | phone, postcode            |
| restaurant | address, area, food, name, pricerange                    | Day, People, choice, day, none, people, phone, reference, time |                      | Ref, Time, postcode        |
| taxi       | departure, destination, taxi_types, type                 | arriveBy, leaveAt, none                                      |                      | phone, taxi_phone          |
| train      | departure, destination                                   | Ref, arriveBy, choice, day, duration, leaveAt, none, people  | price                | reference, ticket, trainID |





#### Machine translation for dialogue data

Machine translated dialogue data using the translated ontology and google translator.

- `[train|val|test].json.zip`



#### Human translation for dialogue data

- `human_val.json.zip`: we sample 250 dialogues from machine translated validation set and ask professional human translators to correct the utterance translation.




