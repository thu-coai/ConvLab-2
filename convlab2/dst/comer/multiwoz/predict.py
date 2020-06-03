import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from convlab2.dst.comer.multiwoz import models
from convlab2.dst.comer.multiwoz import dataloader
from convlab2.dst.comer.multiwoz import utils
from convlab2.dst.comer.multiwoz import dict as dic
from convlab2.dst.comer.multiwoz.preprocess_mw import preprocess_data
from convlab2.dst.comer.multiwoz.comer import revert_state
from convlab2.util.multiwoz.state import default_state
from pprint import pprint
import copy
import os
import argparse
import time
import math
import json
import collections

from collections import defaultdict
#config
parser = argparse.ArgumentParser(description='predict.py')
parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='data/log/norml_mwNestedNOND128NOsaA1FNN/checkpoint.pt', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', action='store_true',
                    help="load pretrain embedding")
parser.add_argument('-limit', type=int, default=0,
                    help="data limit")
parser.add_argument('-log', default='predict', type=str,
                    help="log directory")
parser.add_argument('-unk', action='store_true',
                    help="replace unk")
parser.add_argument('-memory', action='store_true',
                    help="memory efficiency")
parser.add_argument('-beam_size', type=int, default=1,
                    help="beam search size")

opt = parser.parse_args([])
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
use_cuda = True
if use_cuda:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(opt.seed)
#checkpoint
if opt.restore:
    print('loading checkpoint...\n')    
    checkpoints = torch.load(opt.restore)
    
#data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)
vocab = torch.load('data/save_data.tgt.dict')
reversed_vocab = {i:j for j, i in vocab.items()}
print('loading time cost: %.3f' % (time.time()-start_time))

testset = datas['test']
# testset = preprocess_data()
src_vocab, tgt_vocab = vocab, vocab
testloader = dataloader.get_loader(testset, batch_size=1, shuffle=False, num_workers=2)
print(len(testloader))
print(len(testset))
from pytorch_pretrained_bert import BertModel
from convlab2.dst.comer.multiwoz.convert_mw import bert,tokenizer,bert_type

pretrain_embed={}
pretrain_embed['slot'] = torch.load('emb_tgt_mw.pt')
    

# model
print('building model...\n')
bmodel = BertModel.from_pretrained(bert_type)
bmodel.eval()
if use_cuda:
    bmodel.to('cuda')
model = getattr(models, opt.model)(config, src_vocab, tgt_vocab, use_cuda,bmodel,
                       pretrain=pretrain_embed, score_fn=opt.score) 

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]


model.eval()

preds = []
labels = []
joint_preds=[]
joint_labels=[]
joint_allps=[]
joint_alls=[]
    
reference, candidate, source, alignments = [], [], [], []
for src1, src1_len, src2,src2_len, src3, src3_len, tgt, tgt_len,tgtv, tgtv_len,tgtpv, tgtpv_len in testloader:

    if use_cuda:
        src1 = src1.cuda()
        src2 = src2.cuda()
        src3 = src3.cuda()
        tgtpv = tgtpv.cuda()
        src1_len = src1_len.cuda()
        src2_len = src2_len.cuda()
        src3_len = src3_len.cuda()
        tgtpv_len = tgtpv_len.cuda()

        samples,ssamples,vsamples,_ = model.sample(src1, src1_len, src2,src2_len, src3, src3_len,tgtpv, tgtpv_len)
        # get prediction sequence
        for x,xv,xvv,y,yv,yvv in zip(samples,ssamples,vsamples, tgt[0],tgtv[0],tgtpv[0]):
            #each turn
            x=x.data.cpu()
            y=y.data.cpu()
            xt=x[0][:-1].tolist()

            vvt=defaultdict(dict)
            svt={}
#             print("xvv:", xvv)
            for i,k in enumerate(xvv[:-1]):
                slots=xv[:-1][i][0][:-1].tolist()
                if len(slots)!=0:
                    for ji,j in enumerate(k[:-1]):
                        jt=j[0][:-1].tolist()
                        svt[xt[i]]=set(slots)
#                       print("jt:",jt)
                        if len(jt)!=0:
                            vvt[xt[i]][slots[ji]]=jt
            preds.append(set(xt))
            joint_preds.append(svt)
            joint_allps.append(vvt)
            #print("vvt:",vvt)
            pprint(revert_state(vvt, reversed_vocab))


            #print(joint_preds)
            label = []
            for l in y[1:].tolist():
                if l == dic.EOS:
                    break
                label.append(l)

            labels.append(set(label))

            joint_label = {}
            joint_all=defaultdict(dict)
            for i,j in enumerate(yvv[1:].tolist()):
                slots=yv[1:].tolist()[i]
                if sum(slots[1:])==0:
                    break
                else:
                    s=[]
                    for l in slots[1:]:
                        if l == dic.EOS:
                            break
                        s.append(l)
                    if len(s)!=0:
                        joint_label[label[i]]=set(s)
                        for ki,k in enumerate(j[1:]):
                            if sum(k[1:])==0:
                                break
                            else:
                                v=[]
                                for l in k[1:]:
                                    if l == dic.EOS:
                                        break
                                    v.append(l)
                                if len(v)!=0:
                                    joint_all[label[i]][s[ki]]=v
            joint_labels.append(joint_label)
            joint_alls.append(joint_all)

# calculate acc
acc = []
jacc=[]
jaacc=[]
for p,l,jp,jl,jap,jal in zip(preds, labels,joint_preds,joint_labels,joint_allps,joint_alls):
    acc.append(p == l) 
    jacc.append(jp == jl)
    jaacc.append(jap == jal)
acc=sum(acc) / len(acc)
jacc=sum(jacc) / len(jacc)
jaacc=sum(jaacc) / len(jaacc)
print("slot_acc = {}\n".format(acc))      
print("joint_ds_acc = {}\n".format(jacc))
print("joint_all_acc = {}\n".format(jaacc))
  
