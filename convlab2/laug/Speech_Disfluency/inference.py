from .LSTMCRF import BiLSTM_CRF
import json
import numpy as np
import torch
import os
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 100

# Make up some training data
def prepare_sequence(seq, to_ix):
	idxs=[]
	for w in seq:
		if w in to_ix:
			idxs.append(to_ix[w])
		else:
			idxs.append(0)
	return torch.tensor(idxs, dtype=torch.long)

# Put your dir to glove here
glove_file='[dir_to]/glove.6B.100d.txt'

word_to_ix={}
max=20000
ifs=open(glove_file, 'r')
word_to_ix['<unk>'] = 0
weights=[]
weights.append(torch.from_numpy(np.array([0.]*100)))
for i,line in enumerate(ifs.readlines()):
	if i>=max:
		break
	line_list = line.split()
	word = line_list[0]
	embed = line_list[1:]
	embed = torch.from_numpy(np.array([float(num) for num in embed]))
	word_to_ix[word] = i+1
	weights.append(embed)

weights = torch.stack(weights, 0).float()

tag_to_ix = {"O": 0, "F": 1, "R": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM,weights)
model_path=os.path.dirname(os.path.abspath(__file__))
model.load_state_dict(torch.load(os.path.join(model_path,'model/LSTMCRF.bin')))

def IP_model(word_list):
	with torch.no_grad():
		precheck_sent = prepare_sequence(word_list, word_to_ix)
	return model(precheck_sent)[1]

if __name__=="__main__":
	sent="okay , i like to do weight training and cycling ."
	print(IP_model(sent.split()))
