import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from convlab2.dst.comer.multiwoz import dict
from convlab2.dst.comer.multiwoz import models
import time
import numpy as np
from convlab2.dst.comer.multiwoz.convert_mw import bert,tokenizer,bert_type


def norm_emb(emb):
    with torch.no_grad():
        norm = emb.weight.norm(p=2, dim=1, keepdim=True)
        emb.weight = emb.weight.div(norm.expand_as(emb.weight))
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return -(seq_range_expand >= seq_length_expand).float()*1e9

class seq2seq(nn.Module):

    def __init__(self, config, src_vocab, tgt_vocab, use_cuda, bmodel,pretrain=None, score_fn=None):
        super(seq2seq, self).__init__()
        if pretrain is not None:
            self.slot_embedding = nn.Embedding.from_pretrained(pretrain['slot'],freeze=False)
        else:
            self.slot_embedding = None
        if bert:
            self.src_embedding=bmodel
        self.dsep_id=src_vocab['-']
        self.ssep_id=src_vocab[',']
        self.vsep_id=src_vocab[';']
        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)
        
        self.encoder = models.rnn_encoder(config,)# tgt_vocab_size,False, embedding=slot_embedding)
       
        self.decoder = models.rnn_decoder(config, src_vocab_size,tgt_vocab_size, slot_embedding=self.slot_embedding,
                                          value_embedding=self.slot_embedding,score_fn=score_fn)
        
        self.use_cuda = use_cuda
        self.config = config
        self.criterion = models.criterion(tgt_vocab_size, use_cuda)
    
    def forward(self, src1, src1_len, src2, src2_len, src3, src3_len, tgt, tgt_len,tgtv, tgtv_len,tgtpv, tgtpv_len):
        
        b_out, b_state = self.encoder(src3, src3_len,False, embedding=self.slot_embedding)
        user_out, user_state = self.encoder(src1, src1_len,True, embedding=self.src_embedding)
        
        sys_out, sys_state = self.encoder(src2, src2_len,True, embedding=self.src_embedding)

        # combine state
        state = ((b_state[0] + user_state[0] + sys_state[0])/3, (b_state[1] + user_state[1] + sys_state[1])/3)
        
        lengths=(sequence_mask(src3_len,src3.size(0)),sequence_mask(src1_len,src1.size(0)),sequence_mask(src2_len,src2.size(0)))
        story=torch.cat((src3,src2,src1),dim=0).transpose(0,1)
        # decoder
        outputs, loss,num = \
            self.decoder(tgt, state,False,(b_out.transpose(0, 1), \
                     user_out.transpose(0, 1), sys_out.transpose(0, 1)),self.criterion,story,lengths)
        
        soutputs=[]
        voutputs=[]
        vnums=0
        snums=0
        for j,d in enumerate(outputs):#tgt[i].split(1):
            
            soutput, sloss,snum = \
                self.decoder(tgtv[j+1], state,True, (b_out.transpose(0, 1), \
                         user_out.transpose(0, 1), sys_out.transpose(0, 1)),self.criterion,story,lengths,d)
            soutputs.append(sloss)
            snums+=snum
            
            for k,s in enumerate(soutput):
                voutput, vloss,vnum = \
                    self.decoder(tgtpv[j+1,k+1], state,True, (b_out.transpose(0, 1), \
                         user_out.transpose(0, 1), sys_out.transpose(0, 1)),self.criterion,story,lengths,s)
                voutputs.append(vloss)
                vnums+=vnum
        
        return loss,soutputs,voutputs,num,snums,vnums

                      
    def sample(self, src1s, src1_lens, src2s, src2_lens, src3s, src3_lens,tgtpvs, tgtpv_lens):

        # srcs: B x N x T.
        all_sample_ids = []
        all_ssample_ids = []
        all_vsample_ids = []
        bsize=src1s.shape[0]
        BOS=dict.BOS
        EOS=dict.EOS
        bos = torch.ones(bsize).long().fill_(BOS) # [B]
        vp=torch.LongTensor([[BOS],[EOS]]).cuda()
        if self.use_cuda:
            bos = bos.cuda()
        # iterate each turn
        all_hidden=[]
        all_vhidden=[]
        
        for i in range(src1s.shape[1]): 
            # src: B x T -> T x B
            src1 = src1s[:,i,:].transpose(0, 1)
            src2 = src2s[:,i,:].transpose(0, 1)
            src1_len = src1_lens[:,i]
            src2_len = src2_lens[:,i]

            
            if self.use_cuda:
                src1 = src1.cuda()
                src2 = src2.cuda()
                src1_len = src1_len.cuda()
                src2_len = src2_len.cuda()

                
            # first pass use prev label

            if i == 0:
                b_in=vp
            else:
                b_in=belief_ids
            b_len=torch.LongTensor([len(b_in)]).cuda()
                
            # use the output of last turn as previous belief state
            b_out, b_state = self.encoder(b_in,b_len ,False, embedding=self.slot_embedding)
            
            user_out, user_state = self.encoder(src1, src1_len,True, embedding=self.src_embedding)
            sys_out, sys_state = self.encoder(src2, src2_len,True, embedding=self.src_embedding)


            lengths=(sequence_mask(b_len,b_in.size(0)),sequence_mask(src1_len,src1.size(0)),sequence_mask(src2_len,src2.size(0)))
            story=torch.cat((b_in,src2,src1),dim=0).transpose(0,1)
            # combine state
            state = ((b_state[0] + user_state[0] + sys_state[0])/3, (b_state[1] + user_state[1] + sys_state[1])/3)

            # decoder    
            sample_ids, final_outputs = self.decoder.sample(bos, state, False,(b_out.transpose(0, 1), \
                    user_out.transpose(0, 1), sys_out.transpose(0, 1)),story,lengths)
            
            alignments=[]
            all_sample_ids.append(sample_ids.t())#b,s
            
            sids=[]   
            soutputs=[] 
            vids=[]
            belief_ids=[BOS]
            domains=sample_ids.squeeze(1).tolist()
            for j,d in enumerate(final_outputs[0]):#tgt[i].split(1):
                slot_exist=True
                tmp_state=[]
                
                ssample_ids, final_state = \
                    self.decoder.sample(bos, state,True, (b_out.transpose(0, 1), \
                             user_out.transpose(0, 1), sys_out.transpose(0, 1)),story,lengths,d)
                
                sids.append(ssample_ids.t())#(B,S)
                if j < len(domains)-1:
                    tmp_state.append(domains[j])
                    tmp_state.append(self.dsep_id)
                
                vid=[]
                slots=ssample_ids.squeeze(1).tolist()
                if len(slots)==1:
                    slot_exist=False
                cnt=0
                for k,s in enumerate(final_state[0]):#tgt[i].split(1):
                    
                    vsample_ids, vfinal_state = \
                        self.decoder.sample(bos, state,True, (b_out.transpose(0, 1), \
                             user_out.transpose(0, 1), sys_out.transpose(0, 1)),story,lengths,s)
                   
                    vid.append(vsample_ids.t())

                    values=vsample_ids.squeeze(1).tolist()
                    if len(values)==1:
                       cnt+=1
                       continue

                    if k < len(slots)-1 and j < len(domains)-1:
                        tmp_state.append(slots[k])
                        tmp_state.append(self.ssep_id)
                        tmp_state.extend(values[:-1])
                        tmp_state.append(self.vsep_id)
                
                if slot_exist and cnt!=len(slots):
                    belief_ids.extend(tmp_state)    
                vids.append(vid)
            belief_ids.append(EOS)
            #print(belief_ids) 
            belief_ids=torch.LongTensor(belief_ids).unsqueeze(1).cuda()
            
            all_vsample_ids.append(vids)
            all_ssample_ids.append(sids)

        
        return all_sample_ids,all_ssample_ids,all_vsample_ids, alignments


