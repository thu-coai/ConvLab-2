## EDA
Query database and randomly replace some dialog actions.

Randomly replace, delete, swap or insert some words in text.  

## Requirements
It requires `nltk`.
```shell script
pip install nltk
```

And you must download `wordnet` first.

```python
import nltk
nltk.download('wordnet')
```

Database of sgd and frames is available at [Link](http://115.182.62.174:9876/). Please place ```db/``` folder under ```Word_Perturbation/``` dir.


## Run
```shell script
python run.py --multiwoz MULTIWOZ_FILEPATH --output AUGMENTED_MULTIWOZ_FILEPATH
```

Run ```python run.py --help``` for more information about arguments.

