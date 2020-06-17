# CrossWOZ EN

Kaili Huang

This directory contains translated CrossWOZ dataset (from Chinese to English)

#### Ontology translation

- **vocab_dict.json**: translation of the ontology that appear in the data.
- **ontology_translate.py**: translation function that can be used to translate original value using `vocab_dict.json`.

annotation translation are categorized by slot:

| domain     | human translation                     | machine translation      | template translation | refer to other slots' translation                            | don't translate        |
| ---------- | ------------------------------------- | ------------------------ | -------------------- | ------------------------------------------------------------ | ---------------------- |
| Attraction | name, address                         | duration                 | fee, rating          | nearby attract., nearby rest., nearby hotels, selectedResults, source domain | phone                  |
| Restaurant | name, dishes, address                 | open                     | cost, rating         | nearby attract., nearby rest., nearby hotels, selectedResults, source domain | phone                  |
| Hotel      | name, type, Hotel Facilities, address | Hotel Facilities - xxx   | price, rating        | nearby attract., nearby rest., nearby hotels, selectedResults, source domain | phone                  |
| Taxi       |                                       |                          |                      | to, from, selectedResults                                    | car type, plate number |
| Metro      |                                       | from station, to station |                      | to, from, selectedResults                                    |                        |



#### Machine translation for dialogue data

Machine translated dialogue data using the translated ontology and google translator.

- `[train|val|test].json.zip`



#### Human translation for dialogue data

- `human_val.json.zip`: we sample 250 dialogues from machine translated validation set and ask professional human translators to correct the utterance translation.




