import torch
import torch.utils.data as torch_data
import os
class dataset(torch_data.Dataset):

    def __init__(self, src1, src2,src3, tgt, tgtv,tgtpv):#raw_src1, raw_src2, raw_src3, raw_tgt):

        self.src1 = src1
        self.src2 = src2
        self.src3 = src3
        self.tgt = tgt
        self.tgtv = tgtv
        self.tgtpv = tgtpv

    def __getitem__(self, index):
        return self.src1[index], self.src2[index],self.src3[index], self.tgt[index],self.tgtv[index],self.tgtpv[index]#,self.srcv[index]

    def __len__(self):
        return len(self.src1)


def load_dataset(path):
    pass

def save_dataset(dataset, path):
    if not os.path.exists(path):
        os.mkdir(path)

# hierarchical padding

def padding(data):
    # first iteration
    max_ulen,max_slen=128,128
    src1_iter, src2_iter,src3_iter, tgt_iter, tgtv_iter, tgtpv_iter= zip(*data) 
    
    num_dialogue = len(src1_iter)
    
    turn_lens, src1_lens, src2_lens, src3_lens, tgt_lens,tgtv_lens, tgtpv_lens  =[], [], [], [], [], [], []
    
    for src1, src2, src3, tgt,tgtv,tgtpv in zip(src1_iter, src2_iter,src3_iter, tgt_iter, tgtv_iter, tgtpv_iter):
        
        turn_lens.append(len(src1))
        src1_lens.extend([len(s) for s in src1])
        src2_lens.extend([len(s) for s in src2])
        src3_lens.extend([len(s) for s in src3])
        tgt_lens.extend([len(s) for s in tgt])
        tgtv_lens.extend([len(s) for v in tgtv for s in v])  
        tgtpv_lens.extend([len(ss) for v in tgtpv for s in v for ss in s])  
    
    max_turn_len = max(turn_lens)
    max_src1_len = min(max(src1_lens),max_ulen)
    max_src2_len = min(max(src2_lens),max_slen)
    max_src3_len = max(src3_lens)
    max_tgt_len = max(tgt_lens)
    max_tgtv_len = max(tgtv_lens)
    max_tgtpv_len = max(tgtpv_lens)
    # second iteration
    src1_iter, src2_iter,src3_iter, tgt_iter, tgtv_iter, tgtpv_iter= zip(*data) 
    
    
    src1_pad = torch.zeros(num_dialogue, max_turn_len, max_src1_len).long()
    src2_pad = torch.zeros(num_dialogue, max_turn_len, max_src2_len).long()
    src3_pad = torch.zeros(num_dialogue, max_turn_len, max_src3_len).long()
    tgt_pad = torch.zeros(num_dialogue, max_turn_len, max_tgt_len).long()
    tgtv_pad = torch.zeros(num_dialogue, max_turn_len, max_tgt_len,max_tgtv_len).long() 
    tgtpv_pad = torch.zeros(num_dialogue, max_turn_len, max_tgt_len,max_tgtv_len,max_tgtpv_len).long() 
    src1_len = torch.ones(num_dialogue, max_turn_len).long()#full
    src2_len = torch.ones(num_dialogue, max_turn_len).long()#full
    src3_len = torch.ones(num_dialogue, max_turn_len).long()
    tgt_len = torch.ones(num_dialogue, max_turn_len).long()
    tgtv_len = torch.ones(num_dialogue, max_turn_len, max_tgt_len).long() 
    tgtpv_len = torch.ones(num_dialogue, max_turn_len, max_tgt_len,max_tgtv_len).long() 
    raw_src1s, raw_src2s, raw_src3s, raw_tgts = [], [], [], []
       
    dialogue_idx = 0
    for src1, src2, src3, tgt,tgtv,tgtpv in zip(src1_iter, src2_iter,src3_iter, tgt_iter, tgtv_iter, tgtpv_iter):
         
        # user
        for i, s in enumerate(src1): # each turn
            if len(s) > 0: # not null string
                end = min(len(s),max_ulen)
                src1_pad[dialogue_idx, i, :end] = torch.LongTensor(s[:end])
                src1_len[dialogue_idx, i] = end
        # system
        for i, s in enumerate(src2): # each turn
            if len(s) > 0: # not null string
                end = min(len(s),max_slen)
                src2_pad[dialogue_idx, i, :end] = torch.LongTensor(s[:end])
                src2_len[dialogue_idx, i] = end
        # prev label
        for i, s in enumerate(src3):
            if len(s) > 0:
                end = len(s)
                src3_pad[dialogue_idx, i, :end] = torch.LongTensor(s[:end])
                src3_len[dialogue_idx, i] = end
        # label
        for i, s in enumerate(tgt):
            if len(s) > 0:
                end = len(s)
                tgt_pad[dialogue_idx, i, :end] = torch.LongTensor(s[:end])
                tgt_len[dialogue_idx, i] = end
                # label
        for i, s in enumerate(tgtv):
            for j,v in enumerate(s):
                if len(v) > 0:
                    end = len(v)
                    tgtv_pad[dialogue_idx, i,j, :end] = torch.LongTensor(v[:end])
                    tgtv_len[dialogue_idx, i,j] = end
                    
        for i, s in enumerate(tgtpv):
            for j,v in enumerate(s):
                for k,vv in enumerate(v):
                    if len(vv) > 0:
                        end = len(vv)
                        tgtpv_pad[dialogue_idx, i,j,k, :end] = torch.LongTensor(vv[:end])
                        tgtpv_len[dialogue_idx, i,j,k] = end
                
        dialogue_idx += 1    
            
    return  src1_pad, src1_len, \
            src2_pad, src2_len, \
           src3_pad, src3_len, \
           tgt_pad, tgt_len,\
            tgtv_pad, tgtv_len,\
            tgtpv_pad, tgtpv_len 


def get_loader(dataset, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=padding)
    return data_loader
