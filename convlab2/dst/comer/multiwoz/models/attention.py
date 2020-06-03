import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.init as init
import math

class global_attention(nn.Module):

    def __init__(self, hidden_size, activation=None):
        super(global_attention, self).__init__()

        self.linear_in = nn.Linear(hidden_size, hidden_size)
        init.kaiming_normal_(self.linear_in.weight.data)
        init.constant_(self.linear_in.bias.data, 0)
        
        self.softmax = nn.Softmax(dim=1)
        self.activation = activation
        
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        init.kaiming_normal_(self.linear_out.weight.data)
        init.constant_(self.linear_out.bias.data, 0)
        
        
    def forward(self, x, context,mask):
        #context:batch*time*size   
        gamma_h = self.linear_in(x).unsqueeze(2)    # batch * size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)+mask   # batch * time
        
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1) # batch * size
        c_t=self.linear_out(c_t)
        return c_t, weights
       

