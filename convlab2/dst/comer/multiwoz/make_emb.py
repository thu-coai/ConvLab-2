import torch
from convlab2.dst.comer.multiwoz.convert_mw import bert,tokenizer,bert_type
from pytorch_pretrained_bert import BertModel
torch.cuda.set_device(0)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)
bmodel = BertModel.from_pretrained(bert_type)
bmodel.eval()
bmodel.to('cuda')

tgtD=torch.load('data/save_data.tgt.dict')
emb=[]
itl={i:v for (v,i) in tgtD.items()}
for i in range(len(tgtD)):
    label = itl[i]
    x1=tokenizer.convert_tokens_to_ids(label.split())
    if i > len(tgtD)-5:
        print(label)
        print(x1)
    encoded_layers, _ =bmodel(torch.LongTensor(x1).cuda().unsqueeze(0),token_type_ids=None, attention_mask=None)   
    x=torch.stack(encoded_layers,-1).mean(-1).mean(-2)
    emb.append(x.detach().cpu())
x=torch.cat(emb,0)
torch.save(x,'emb_tgt_mw.pt')
print(x.shape)
print(x.numpy())
