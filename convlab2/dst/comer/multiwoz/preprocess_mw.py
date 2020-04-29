import argparse
import torch
from convlab2.dst.comer.multiwoz.dataloader import dataset
import json
import os
parser = argparse.ArgumentParser(description='preprocess.py')
from convlab2.dst.comer.multiwoz.convert_mw import bert,tokenizer
##
## **Preprocess Options**
##

parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-train_tgt', 
                    default='data/mwoz2_format_train.json',
                    help="Path to the training target data")
parser.add_argument('-valid_tgt', 
                    default='data/mwoz2_format_dev.json',
                    help="Path to the validation target data")
parser.add_argument('-test_tgt', 
                    default='data/mwoz2_format_test.json',
                    help="Path to the validation target data")

parser.add_argument('-save_data', 
                    default='data/save_data',
                    help="Output file for the prepared data")

parser.add_argument('-shuffle',    type=int, default=0,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")


opt = parser.parse_args()
torch.manual_seed(opt.seed)


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    torch.save(vocab,file)
 

def makeData(srcFile, tgtDicts):
    
    src1, src2, src3, tgt,srcv, tgtv = [], [], [], [], [],[]
    count, ignored = 0, 0

    print('Processing %s  ...' % (srcFile))
    srcF = open(srcFile, 'r')
    for l in srcF: # for each dialogue
        l = eval(l) 
        src1_tmp, src2_tmp, src3_tmp, tgt_tmp,tgt_vtmp,src_vtmp = [], [], [], [],[],[]

        # hierarchical input for a whole dialogue with multiple turns
        slines = l['system_input']
        ulines = l['user_input']
        plines = l['belief_input']
        pvlines = l['labeld']
        tlines = l['labels']
        tvlines = l['labelv']  

        for sWords, uWords, pWords, tWords,tvWords,pvWords in zip(slines, ulines, plines, tlines,tvlines,pvlines):  
            

            # src vocab
            if bert:
                src1_tmp += [[tgtDicts[w] for w in uWords]] 
                src2_tmp += [[tgtDicts[w] for w in sWords]] 
            # tgt vocab
            src3_tmp += [[tgtDicts[w] for w in pWords]]
            tt=[tgtDicts[w] for w in pvWords]
            tgt_tmp += [tt]
            tv=[[tgtDicts[w] for w in ws] for ws in tWords]
            tgt_vtmp +=[tv]
           
            tpv=[[[tgtDicts[w] for w in ws] for ws in wss] for wss in tvWords]
            src_vtmp +=[tpv]            
            

        count += 1

        src1.append(src1_tmp)
        src2.append(src2_tmp)
        src3.append(src3_tmp)
        srcv.append(src_vtmp)
        tgt.append(tgt_tmp)
        tgtv.append(tgt_vtmp)

        
    srcF.close()
    print(srcv[:5])

    print('Prepared %d dialogues' %
          (len(src1)))

    return dataset(src1, src2,src3, tgt,tgtv,srcv)

def preprocess_data():
    # path = 'data/'
    # sl_dict = json.load(open(os.path.join(path, 'mwoz2_sl.dict')))
    # dm_dict = json.load(open(os.path.join(path, 'mwoz2_dm.dict')))
    # dicts = {}
    # dicts['src'] = tokenizer.vocab
    #
    # print(len(dicts['src']))
    #
    # for j in sl_dict.keys():
    #     for i in sl_dict[j].keys():
    #         if i not in dicts['src'].keys():
    #             dicts['src'][i] = len(dicts['src'])
    #             print(i)
    #             print(dicts['src'][i])
    #
    # for i in dm_dict.keys():
    #     if i not in dicts['src'].keys():
    #         dicts['src'][i] = len(dicts['src'])
    #         print(i)
    #         print(dicts['src'][i])
    #
    # print(len(dicts['src']))
    save_data = torch.load(opt.save_data)
    dicts = save_data['dicts']
    print(len(dicts['src']))
    vocab = dicts['src']
    json.dump(vocab, open('data/vocab.json', 'w'))
    real = makeData('data/mwoz2_format_real.json', dicts['src'])
    return real


def main():
    preprocess_data()
    # # preprocess data and store as .pt file
    # dicts = {}
    # dicts['src'] = tokenizer.vocab
    #
    # path='data/mwoz2_sl.dict'
    # srcF = open(path, 'r')
    #
    # sl_dict=json.loads(srcF.read())
    # srcF.close()
    #
    # path='data/mwoz2_dm.dict'
    # srcF = open(path, 'r')
    # dm_dict=json.loads(srcF.read())
    # srcF.close()
    #
    # print(len(dicts['src']))
    #
    # print(sl_dict)
    # print(dm_dict)
    #
    # for j in sl_dict.keys():
    #     for i in sl_dict[j].keys():
    #         if i not in dicts['src'].keys():
    #             dicts['src'][i]=len(dicts['src'])
    #             print(i)
    #             print(dicts['src'][i])
    #
    # for i in dm_dict.keys():
    #     if i not in dicts['src'].keys():
    #         dicts['src'][i]=len(dicts['src'])
    #         print(i)
    #         print(dicts['src'][i])
    #
    #
    # print(len(dicts['src']))
    #
    # # src/tgt are the same file
    # # train/valid/test should be in hierarchical structure
    # print('Preparing training ...')
    # train = makeData(opt.train_tgt,  dicts['src'])
    #
    # print('Preparing validation ...')
    # valid = makeData(opt.valid_tgt, dicts['src'])
    #
    # print('Preparing test ...')
    # test = makeData(opt.test_tgt, dicts['src'])
    #
    # saveVocabulary('target', dicts['src'], opt.save_data + '.tgt.dict')
    #
    # print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    # save_data = {'dicts': dicts,
    #              'train': train,
    #              'valid': valid,
    #              'test': test}
    # torch.save(save_data, opt.save_data)

if __name__ == "__main__":
    print(opt)
    main()
