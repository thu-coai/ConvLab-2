## Speech_Disfluency

The interruption points are predictedby a Bi-LSTM+CRF model. 

The fillerwords, restart terms, and edit terms and their occurrence frequency are all sampled from their distribution in SwitchBoard.


## Bi-LSTM+CRF model

Bi-LSTM+CRF model is trained on SwitchBoard data.

Please download the pre-trained parameters and disfluency resources at [Link](http://115.182.62.174:9876/).

The model requires glove.6B.100d wordvector, please modify line22 in inference.py.
