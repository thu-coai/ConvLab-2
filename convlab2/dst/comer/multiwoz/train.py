import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

from convlab2.dst.comer.multiwoz import models
from convlab2.dst.comer.multiwoz import dataloader
from convlab2.dst.comer.multiwoz import utils
from convlab2.dst.comer.multiwoz import dict as dic
import torch.nn.init as init
import os
import argparse
import time
import collections
from pytorch_pretrained_bert import BertModel
from convlab2.dst.comer.multiwoz.convert_mw import bert,tokenizer,bert_type
from collections import defaultdict
#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', default=True, type=bool,
                    help="load pretrain embedding")
parser.add_argument('-notrain', default=False, type=bool,
                    help="train or not")
parser.add_argument('-limit', default=0, type=int,
                    help="data limit")
parser.add_argument('-log', default='temp', type=str,
                    help="log directory")
parser.add_argument('-unk', default=True, type=bool,
                    help="replace unk")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0

if use_cuda:
    print(opt.gpus)
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
print(use_cuda)
if opt.restore: 
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore,map_location="cuda:0")


# data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)  
print('loading time cost: %.3f' % (time.time()-start_time))

trainset, validset = datas['train'], datas['valid']
testset = datas['test']
src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['src']

print(len(tgt_vocab))
trainloader = dataloader.get_loader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)
# consider batch_size=1 for valid/test
validloader = dataloader.get_loader(validset, batch_size=1, shuffle=False, num_workers=2)
testloader = dataloader.get_loader(testset, batch_size=1, shuffle=False, num_workers=2)

if opt.pretrain:
    pretrain_embed={}
    pretrain_embed['slot'] = torch.load('emb_tgt_mw.pt')
    
else:
    pretrain_embed = None

# model
print('building model...\n')
bmodel = BertModel.from_pretrained(bert_type)
bmodel.eval()
if use_cuda:
    with torch.no_grad():
        bmodel.cuda()
model = getattr(models, opt.model)(config, src_vocab, tgt_vocab, use_cuda,bmodel,
                       pretrain=pretrain_embed, score_fn=opt.score) 


        
# load checkpoint - (continue training)
if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
#This version does not support distributed/parallel training.
#if len(opt.gpus) > 1:  
#    model = nn.DataParallel(model, device_ids=opt.gpus, dim=0)

  
params={n:p for n,p in model.named_parameters() if 'src_embedding' not in n }
    
# optimizer
if opt.restore:
    optim = checkpoints['optim']
else: 
    optim = torch.optim.Adam(params.values(), lr=config.learning_rate,amsgrad=True)


# total number of parameters
param_count = 0
for param in params.values():
    param_count += param.view(-1).size()[0]

if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'log.txt') 
logging_csv = utils.logging_csv(log_path+'record.csv') 
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")  

logging('total number of trainable parameters: %d\n\n' % param_count)
logging('score function is %s\n\n' % opt.score)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

total_loss, start_time = 0, time.time()
report_total, report_correct = 0, 0
report_vocab, report_tot_vocab = 0, 0
best_acc=0
best_jacc=0
best_jaacc=0
best_epoch=0
# train
def train_and_eval(epoch):
    global e, updates, total_loss, start_time, report_total,best_acc,best_jacc,best_jaacc,best_epoch


    e = epoch
    model.train()
    
    total_loss = 0
    total_sloss = 0
    start_time = time.time()
    report_total = 0
    total_vloss = 0
    
    app_time=[]


    for src1, src1_len, src2,src2_len, src3, src3_len, tgt, tgt_len,tgtv, tgtv_len,tgtpv, tgtpv_len in trainloader:
        if use_cuda:
            src1 = src1.cuda()#user
            src2 = src2.cuda()
            src3 = src3.cuda()#belief
            tgt = tgt.cuda()#domain
            tgtv = tgtv.cuda()#slot
            tgtpv = tgtpv.cuda()#value
            src1_len = src1_len.cuda()
            src2_len = src2_len.cuda()
            src3_len = src3_len.cuda()
            tgt_len = tgt_len.cuda()
            tgtv_len = tgtv_len.cuda()
            tgtpv_len = tgtpv_len.cuda()

        total_lossl=[]
        total_slossl=[]
        total_vlossl=[]

        optim.zero_grad()
        for i in range(src1.shape[1]): 
            # batch second
            src1s = src1[:,i,:].transpose(0, 1)
            src2s = src2[:,i,:].transpose(0, 1)
            src3s = src3[:,i,:].transpose(0, 1)
            tgts = tgt[:,i,:].transpose(0, 1)
            tgtvs = tgtv[:,i,:,:].transpose(0, 1).transpose(1, 2)# batch third
            tgtpvs = tgtpv[:,i,:,:,:].transpose(0, 1).transpose(1, 2).transpose(2, 3)
            src1_lens = src1_len[:,i]
            src2_lens = src2_len[:,i]
            src3_lens = src3_len[:,i]
            tgt_lens = tgt_len[:,i]
            tgtv_lens = tgtv_len[:,i,:]
            tgtpv_lens = tgtpv_len[:,i,:,:]


            loss,sloss,vloss,num,snum,vnum = model(src1s, src1_lens, src2s,src2_lens, src3s, src3_lens, tgts, tgt_lens,tgtvs, tgtv_lens,tgtpvs, tgtpv_lens)
        
        
            losses =(loss+sum(sloss)+sum(vloss))/(num+snum+vnum)
            losses=losses/src1.shape[1]
            total_lossl.append(loss.data/num)
            total_slossl.append(sum(sloss).data/snum) 
            total_vlossl.append(sum(vloss).data/vnum) 
            losses.backward()
        
        torch.nn.utils.clip_grad_norm_(params.values(), config.max_grad_norm)
        optim.step()
        optim.zero_grad()
        total_vloss += sum(total_vlossl)/len(total_vlossl)  
        total_sloss += sum(total_slossl)/len(total_slossl)  
        total_loss += sum(total_lossl)/len(total_lossl)  
        
        
        report_total += 1       

        updates += 1  
        
        if updates % config.log_interval==0:
            logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.5f, train sloss: %6.5f, train vloss: %6.5f\n"
                    % (time.time()-start_time, epoch, updates, total_loss / report_total,total_sloss / report_total, total_vloss / report_total))
            total_loss = 0
            total_sloss = 0
            start_time = time.time()
            report_total = 0
            total_vloss = 0
            app_time=[]

    if epoch>4: 
        start_time = time.time()
        logging('========evaluating after %d epochs========\n' % epoch)


        acc,jacc,jaacc = eval(epoch)
        best_acc=max(best_acc,acc)
        best_jacc=max(best_jacc,jacc)
        if jaacc>best_jaacc:
            best_jaacc=jaacc
            best_epoch=epoch
            save_model(log_path+'checkpoint.pt')
        logging("best_slot_acc = {}\n".format(best_acc))   
        logging("best_joint_acc = {}\n".format(best_jacc)) 
        logging("best_joint_all_acc = {} ".format(best_jaacc)) 
        logging("at epoch {}\n".format(best_epoch)) 
        logging("time: %6.3f\n"
                    % (time.time()-start_time))
        logging('==========================================\n')
        model.train()

def eval(epoch):
    model.eval()
    
    preds = []
    labels = []
    joint_preds=[]
    joint_labels=[]
    joint_allps=[]
    joint_alls=[]
    
    reference, candidate, source, alignments = [], [], [], []
    with torch.no_grad():
        for src1, src1_len, src2,src2_len, src3, src3_len, tgt, tgt_len,tgtv, tgtv_len,tgtpv, tgtpv_len in validloader:

            if use_cuda:
                src1 = src1.cuda()
                src2 = src2.cuda()
                src3 = src3.cuda()
                tgtpv = tgtpv.cuda()
                src1_len = src1_len.cuda()
                src2_len = src2_len.cuda()
                src3_len = src3_len.cuda()
                tgtpv_len = tgtpv_len.cuda()

            # get prediction sequence    
            samples,ssamples,vsamples,_ = model.sample(src1, src1_len, src2,src2_len, src3, src3_len,tgtpv, tgtpv_len)
            
            for x,xv,xvv,y,yv,yvv in zip(samples,ssamples,vsamples, tgt[0],tgtv[0],tgtpv[0]):
                #each turn
                x=x.data.cpu()
                y=y.data.cpu()
                
                xt=x[0][:-1].tolist()

                vvt=defaultdict(dict)
                svt={}
#                 print("xvv:", xvv)
                for i,k in enumerate(xvv[:-1]):
                    slots=xv[:-1][i][0][:-1].tolist()
                    if len(slots)!=0:
                        for ji,j in enumerate(k[:-1]):
                            jt=j[0][:-1].tolist()
                            svt[xt[i]]=set(slots)
#                           print("jt:",jt)
                            if len(jt)!=0:
                                vvt[xt[i]][slots[ji]]=jt
                preds.append(set(xt))  
                joint_preds.append(svt)
                joint_allps.append(vvt)
                #print("vvt:",vvt)


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
    logging("slot_acc = {}\n".format(acc))      
    logging("joint_ds_acc = {}\n".format(jacc))
    logging("joint_all_acc = {}\n".format(jaacc))

    return acc,jacc,jaacc


def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'updates': updates,
            'optim': optim,}
    torch.save(checkpoints, path)


def main():

    for i in range(1, config.epoch+1):
        if not opt.notrain:
            train_and_eval(i)
        else:
            eval(i)


if __name__ == '__main__':
    main()
