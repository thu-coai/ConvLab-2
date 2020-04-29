import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from convlab2.dst.comer.multiwoz import dict
from convlab2.dst.comer.multiwoz import models
import time
import numpy as np
import torch.nn.init as init

import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# concat outputs of all layers
class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class rnn_encoder(nn.Module):
    def __init__(self, config, vocab_size=None,):
        super(rnn_encoder, self).__init__()
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirec)
        for param in self.rnn.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.constant_(param.data, 0)
        self.config = config

    def forward(self, x1, x1_len,bert,app_time=None, embedding=None):
        # sort x1
        tot_len=x1.size(0)
        x1_sort_idx = torch.argsort(-1 * x1_len)
        x1_unsort_idx = torch.argsort(x1_sort_idx)
        x1 = x1[:, x1_sort_idx]
        x1_len = x1_len[x1_sort_idx]
        if bert:
            with torch.no_grad():
                max_len=len(x1)
                input_mask=torch.arange(max_len).cuda().expand(len(x1_len), max_len) < x1_len.unsqueeze(1)
                xt=x1.transpose(0, 1)
                encoded_layers, _ =embedding(xt,token_type_ids=None, attention_mask=input_mask)   
                x=torch.stack(encoded_layers,-1).mean(-1).transpose(0, 1)
        else:
            x=embedding(x1)
        embs = pack(x, x1_len)
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs,total_length=tot_len)[0]
       
        # unsort
        outputs = outputs[:, x1_unsort_idx, :] # index batch axis

        if not self.config.bidirec:
            return outputs, (h, c)
        else:
            batch_size = h.size(1)
            h = h.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            c = c.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            state = (h.transpose(0, 1), c.transpose(0, 1))
            return outputs, state

class rnn_decoder(nn.Module):
    def __init__(self, config,src_vocab_size,tgt_vocab_size,slot_embedding=None,value_embedding=None, score_fn=None):
        super(rnn_decoder, self).__init__()
        
        self.slot_embedding=slot_embedding
        self.vocab_size=tgt_vocab_size
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        for param in self.rnn.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.constant_(param.data, 0)   
        self.score_fn = score_fn
        self.slot_linear=nn.Linear(config.decoder_hidden_size, config.emb_size)
        init.kaiming_normal_(self.slot_linear.weight.data)
        init.constant_(self.slot_linear.bias.data, 0)
        
        activation = None

        self.attention = models.global_attention(config.decoder_hidden_size, activation)
        #self.user_attention = models.global_attention(config.decoder_hidden_size, activation)
        #self.sys_attention = models.global_attention(config.decoder_hidden_size, activation)
        self.hidden_size = config.decoder_hidden_size
     
        self.config = config
        
        self.linear_out = nn.Linear(4*self.hidden_size, self.hidden_size)
        init.kaiming_normal_(self.linear_out.weight.data)
        init.constant_(self.linear_out.bias.data, 0)
        self.re1 = nn.ReLU()
#         self.dp1=nn.Dropout(0.1)
        self.linear_slot =nn.Linear(self.hidden_size, self.hidden_size)
        init.kaiming_normal_(self.linear_slot.weight.data)
        init.constant_(self.linear_slot.bias.data, 0)
        self.re2 = nn.ReLU()
#         self.dp2=nn.Dropout(0.1)
        
        self.linear3 =nn.Linear(self.hidden_size, self.hidden_size)
        init.kaiming_normal_(self.linear3.weight.data)
        init.constant_(self.linear3.bias.data, 0)
        self.re3 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.linear4 =nn.Linear(self.hidden_size, self.hidden_size)
        init.kaiming_normal_(self.linear4.weight.data)
        init.constant_(self.linear4.bias.data, 0)
        self.re4 = nn.ReLU()      
        self.dropout = nn.Dropout(0.5)#config.dropout)

    def forward(self, tgt, init_state, ifslot,contexts,criterion,story,lengths,slot=None):
        # context - a tuple of outputs from multiple encoders 
        #sys:2 user:1 belief:0
        inputs=tgt[:-1]
        targets=tgt[1:]
        
        embs = self.slot_embedding(inputs)#S,B,H
        outputs, state, attn1s, attn2s, attn3s = [], init_state, [], [], []
        scores=[]
        # iterate each slot in inputs
        for emb in embs.split(1):
            output0, state = self.rnn(emb.squeeze(0), state) #(B,H) 
            if ifslot:
                outputt=output0+slot#torch.cat([output,slot],-1)
            else:
                outputt=output0
        
            output1, _ = self.attention(outputt, contexts[2],lengths[2])
            outputt1=outputt+output1
            output2, _ = self.attention(outputt1, contexts[1],lengths[1])
            outputt2=outputt1+output2
            output3, _ = self.attention(outputt2, contexts[0],lengths[0])
            outputt3=outputt2+output3
            with torch.no_grad():
                cout=Variable(outputt)
                cout1=Variable(outputt1)
                cout2=Variable(outputt2)

            output = self.re1(self.linear_out(torch.cat([cout, cout1, cout2, outputt3], 1))) 
#             output=self.dp1(output)
            output=self.re2(self.linear_slot(output))
#             output=self.dp2(output)
            output=self.re3(self.linear3(output))
#             output=self.dp3(output)
            output=self.re4(self.linear4(output))
            output = self.dropout(output)
            outputs += [output]
            score=self.compute_score(output,output0,contexts,story,lengths)
            scores.append(score)
        outputs = torch.stack(outputs)#S,B,H
        scores = torch.cat(scores)#S*B,H
        loss = criterion(scores, targets.contiguous().view(-1)) 

        num_total = targets.ne(dict.PAD).data.sum()
        
        return outputs, loss,num_total
        
        # convert hidden to vocab
    def compute_score(self, hiddens,eb,cont,story,lengths):      
        #hiddens = self.dropout(hiddens)
        
        word = self.slot_linear(hiddens)
        embs=self.slot_embedding.weight.t()
        scores_v = torch.matmul(word, embs)          
        scores=self.log_softmax(scores_v)
        return scores 

    # sample all steps - by running sample_one
    def sample(self, input, init_state,ifslot, contexts,story,lengths,slot=None):
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attn1s, attn2s, attn3s = [], [], []
        scores=[]
        inputs += [input]
        soft_score = None
        mask = None
        eos=dict.EOS
        
        #while True:
        for _ in range(20):#max_gen_lengtuh
            # use last output as input
            hidden, state,output, attn1_weights, attn2_weights, attn3_weights = self.sample_one(inputs[-1], state, contexts,ifslot,story,lengths,slot)
            
            
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [hidden]#pass hidden
#             attn1s += [attn1_weights.data.cpu()]
#             attn2s += [attn2_weights.data.cpu()]
#             attn3s += [attn3_weights.data.cpu()]
       
            if predicted == eos:
                break
                    
        sample_ids = torch.stack(sample_ids)#(s*B)
        outputs = torch.stack(outputs)#(s*B*512)
#         attn1s = torch.stack(attn1s)
#         attn2s = torch.stack(attn2s)
#         attn3s = torch.stack(attn3s)
            
        return sample_ids, (outputs, attn1s, attn2s, attn3s)

    # only sample one time step
    def sample_one(self, input, state, contexts,ifslot,story,lengths,slot=None):
        emb = self.slot_embedding(input)#B,H

        output0, state = self.rnn(emb, state)
        if ifslot==True:
            outputt=output0+slot#torch.cat([output,slot],-1)
        else:
            outputt=output0
        
        output1, att1 = self.attention(outputt, contexts[2],lengths[2])
        outputt1=outputt+output1
        output2, att2 = self.attention(outputt1, contexts[1],lengths[1])
        outputt2=outputt1+output2
        output3, att3 = self.attention(outputt2, contexts[0],lengths[0])
        outputt3=outputt2+output3

        output = self.re1(self.linear_out(torch.cat([outputt, outputt1, outputt2, outputt3], 1)))                      
               
        output=self.re2(self.linear_slot(output))
        output=self.re3(self.linear3(output))
        output=self.re4(self.linear4(output))
        score=self.compute_score(output,output0,contexts,story,lengths)
        return output, state,score, att1, att2, att3 #pass hidden out
