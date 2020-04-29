import argparse
import torch
import os
from pprint import pprint
import copy
import json
from collections import defaultdict
from convlab2.dst.dst import DST
from convlab2.util.multiwoz.state import default_state
from convlab2.dst.comer.multiwoz import models
from convlab2.dst.comer.multiwoz import dataloader
from convlab2.dst.comer.multiwoz import utils
from pytorch_pretrained_bert import BertModel, BertTokenizer
from convlab2.dst.comer.multiwoz.create_data import normalize
from convlab2.dst.comer.multiwoz.dataloader import dataset
from convlab2.util.file_util import cached_path
import zipfile


def revert_state(model_output:dict, reversed_vocab:dict):
    slotdict = {'price range':'pricerange', 'leave at':'leaveAt', 'arrive by':'arriveBy'}
    valuedict = {'do not care': 'dontcare'}
    belief_state = default_state()['belief_state']
    for d in model_output:
        for s in model_output[d]:
            for v in model_output[d][s]:
                domain = reversed_vocab[d]
                if domain not in belief_state:
                    continue
                slot = reversed_vocab[s]
                value = reversed_vocab[v]
                if slot.startswith('book '):
                    slot = slot[5:]
                    table = belief_state[domain]['book']
                else:
                    table = belief_state[domain]['semi']
                if slot in slotdict:
                    slot = slotdict[slot]
                if value in valuedict:
                    value = valuedict[value]
                if slot in table:
                    table[slot] = value
    return belief_state


class ComerTracker(DST):
    def __init__(self, model_file='https://convlab.blob.core.windows.net/convlab-2/comer.zip',
                 embed_file='https://convlab.blob.core.windows.net/convlab-2/comer_embed.zip'):
        super().__init__()
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

        self.root_path = os.path.dirname(os.path.abspath(__file__))
        opt = parser.parse_args([])
        config = utils.read_config(os.path.join(self.root_path, opt.config))
        torch.manual_seed(opt.seed)
        use_cuda = True
        bert_type = 'bert-large-uncased'

        if os.path.exists(os.path.join(self.root_path, 'data/mwoz2_dm.dict')) and \
                os.path.exists(os.path.join(self.root_path, 'data/mwoz2_sl.dict')) and \
                os.path.exists(os.path.join(self.root_path, 'data/save_data.tgt.dict')) and \
                os.path.exists(os.path.join(self.root_path, 'data/log/norml_mwNestedNOND128NOsaA1FNN/checkpoint.pt')):
            pass
        else:
            output_dir = os.path.join(self.root_path, 'data')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(self.root_path)
            archive.close()
        if not os.path.exists(os.path.join(self.root_path, 'data/emb_tgt_mw.pt')):
            output_dir = os.path.join(self.root_path, 'data')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from embed_file param')
            archive_file = cached_path(embed_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(self.root_path)
            archive.close()

        self.sl_dict = json.load(open(os.path.join(self.root_path, 'data/mwoz2_sl.dict')))
        self.dm_dict = json.load(open(os.path.join(self.root_path, 'data/mwoz2_dm.dict')))
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.vocab = torch.load(os.path.join(self.root_path, 'data/save_data.tgt.dict'))
        self.reversed_vocab = {i: j for j, i in self.vocab.items()}

        pretrain_embed = {}
        pretrain_embed['slot'] = torch.load(os.path.join(self.root_path, 'data/emb_tgt_mw.pt'))

        print('building model...\n')
        bmodel = BertModel.from_pretrained(bert_type)
        bmodel.eval()
        if use_cuda:
            bmodel.to('cuda')
        self.model = getattr(models, opt.model)(config, self.vocab, self.vocab, use_cuda, bmodel,
                                                pretrain=pretrain_embed, score_fn=opt.score)

        print('loading checkpoint...\n')
        print(os.path.join(self.root_path, opt.restore))
        import sys
        sys.path.append(self.root_path)
        checkpoints = torch.load(os.path.join(self.root_path, opt.restore))
        self.model.load_state_dict(checkpoints['model'])
        self.model.cuda()
        self.model.eval()

        self.state = None
        self.init_session()

    def init_session(self):
        self.state = default_state()

    def update(self, user_act=None):
        if not isinstance(user_act, str):
            raise Exception('Expected user_act is str but found {}'.format(type(user_act)))

        # history = copy.deepcopy(new_state['history'])
        # history[-1].append(user_act) # add current usr
        # history[0][0] = "" # first sen sys
        history = []
        for name, utt in self.state['history']:
            if len(history)==0 or len(history[-1])==2:
                history.append([utt])
            else:
                history[-1].append(utt)
        # print('inside history', history) # [[sys, usr], [sys, usr],...]
        self.state['belief_state'] = self.model_predict(history)
        return self.state

    def model_predict(self, history):
        sdict = {'user_input': [],
                 'system_input': [],
                 'belief_input': [],
                 'labeld': [],
                 'labels': [],
                 'labelv': [],
                 }
        for i, s in enumerate(history):
            ta = ['[CLS]', 'system', ':'] + self.tokenizer.tokenize(normalize(s[0], False)) + ['[SEP]']
            tb = ['[CLS]', 'user', ':'] + self.tokenizer.tokenize(normalize(s[1], False)) + ['[SEP]']
            sdict['system_input'].append(ta)
            sdict['user_input'].append(tb)
            sdict['belief_input'].append([])
            sdict['labeld'].append([])
            sdict['labels'].append([[]])
            sdict['labelv'].append([[[], []]])
        print(sdict)

        src1_tmp, src2_tmp, src3_tmp, tgt_tmp, tgt_vtmp, src_vtmp = [], [], [], [], [], []

        # hierarchical input for a whole dialogue with multiple turns
        slines = sdict['system_input']
        ulines = sdict['user_input']
        plines = sdict['belief_input']
        pvlines = sdict['labeld']
        tlines = sdict['labels']
        tvlines = sdict['labelv']

        for sWords, uWords, pWords, tWords, tvWords, pvWords in zip(slines, ulines, plines, tlines, tvlines, pvlines):
            # src vocab
            src1_tmp += [[self.vocab[w] for w in uWords]]
            src2_tmp += [[self.vocab[w] for w in sWords]]
            # tgt vocab
            src3_tmp += [[self.vocab[w] for w in pWords]]
            tt = [self.vocab[w] for w in pvWords]
            tgt_tmp += [tt]
            tv = [[self.vocab[w] for w in ws] for ws in tWords]
            tgt_vtmp += [tv]

            tpv = [[[self.vocab[w] for w in ws] for ws in wss] for wss in tvWords]
            src_vtmp += [tpv]

        src1, src2, src3, tgt, srcv, tgtv = [src1_tmp], [src2_tmp], [src3_tmp], [tgt_tmp], [src_vtmp], [tgt_vtmp]

        testset = dataset(src1, src2, src3, tgt, tgtv, srcv)
        testloader = dataloader.get_loader(testset, batch_size=1, shuffle=False, num_workers=1)

        for src1, src1_len, src2, src2_len, src3, src3_len, tgt, tgt_len, tgtv, tgtv_len, tgtpv, tgtpv_len in testloader:
            src1 = src1.cuda()
            src2 = src2.cuda()
            src1_len = src1_len.cuda()
            src2_len = src2_len.cuda()

            samples, ssamples, vsamples, _ = self.model.sample(src1, src1_len, src2, src2_len, None, None, None, None)
            # get prediction sequence
            for x, xv, xvv in zip(samples, ssamples, vsamples):
                # each turn
                x = x.data.cpu()
                xt = x[0][:-1].tolist()

                vvt = defaultdict(dict)
                for i, k in enumerate(xvv[:-1]):
                    slots = xv[:-1][i][0][:-1].tolist()
                    if len(slots) != 0:
                        for ji, j in enumerate(k[:-1]):
                            jt = j[0][:-1].tolist()
                            if len(jt) != 0:
                                vvt[xt[i]][slots[ji]] = jt
                model_output = vvt

        new_belief_state = revert_state(model_output, self.reversed_vocab)
        return new_belief_state


def test_update():
    # lower case, tokenized.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tracker = ComerTracker()
    tracker.init_session()
    # original usage in Convlab
    # tracker.state['history'] = [
    #     ["null", "am looking for a place to to stay that has cheap price range it should be in a type of hotel"],
    #     ["Okay, do you have a specific area you want to stay in?", "no, i just need to make sure it's cheap. oh, and i need parking"],
    #     ["I found 1 cheap hotel for you that includes parking. Do you like me to book it?", "Yes, please. 6 people 3 nights starting on tuesday."],
    #     ["I am sorry but I wasn't able to book that for you for Tuesday. Is there another day you would like to stay or perhaps a shorter stay?", "how about only 2 nights."],
    #     ["Booking was successful.\nReference number is : 7GAWK763. Anything else I can do for you?"]
    # ]

    # current usage in Convlab2
    tracker.state['history'] = [
        ['sys', ''],
        ['user', 'Could you book a 4 stars hotel for one night, 1 person?'],
        ['sys', 'If you\'d like something cheap, I recommend the Allenbell']
    ]
    tracker.state['history'].append(['user', 'Friday and Can you book it for me and get a reference number ?'])

    user_utt = 'Friday and Can you book it for me and get a reference number ?'
    from timeit import default_timer as timer
    start = timer()
    pprint(tracker.update(user_utt))
    end = timer()
    print(end - start)

    start = timer()
    tracker.update(user_utt)
    end = timer()
    print(end - start)

    start = timer()
    tracker.update(user_utt)
    end = timer()
    print(end - start)

if __name__ == '__main__':
    test_update()
