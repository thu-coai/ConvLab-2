import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import json
from LSTMCRF2 import BiLSTM_CRF
import numpy as np
from progressbar import progressbar




def prepare_sequence(seq, to_ix):
	idxs=[]
	for w in seq:
		if w in to_ix:
			idxs.append(to_ix[w])
		else:
			idxs.append(0)
	return torch.tensor(idxs, dtype=torch.long)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 100

# Make up some training data



data=json.load(open('SWBD/data.json','r'))
training_data=[]
for d in data:
	training_data.append((d['text'],d['tags']))
print(len(training_data))
glove_file=''

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

model = BiLSTM_CRF( len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM,weights)
model

optimizer = optim.Adam(model.parameters(), lr=0.001)


with torch.no_grad():
	precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
	precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
	print(model(precheck_sent))

ep=0
for epoch in range(30): 
	n,losses=0,0.
	ep+=1
	for sentence, tags in progressbar(training_data):
		model.zero_grad()
		sentence_in = prepare_sequence(sentence, word_to_ix)
		targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
		loss = model.neg_log_likelihood(sentence_in, targets)
		losses+=loss
		n+=1
		loss.backward()
		optimizer.step()
	torch.save(model.state_dict(), 'model/LSTMCRF_'+str(ep)+'.bin')
	print('loss:'+str(losses/n))
	with torch.no_grad():
		precheck_sent = prepare_sequence("okay , i like to do , weight training and cycling .".split(), word_to_ix)
		print(model(precheck_sent))
		precheck_sent = prepare_sequence(training_data[1][0], word_to_ix)
		print(model(precheck_sent))
		precheck_sent = prepare_sequence('i want to go to cambridge .'.split(), word_to_ix)
		print(model(precheck_sent))