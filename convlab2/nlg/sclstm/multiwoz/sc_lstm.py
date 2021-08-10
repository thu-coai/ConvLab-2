import configparser
import os
import zipfile
from copy import deepcopy

import torch

from convlab2.util.file_util import cached_path
from convlab2.nlg.sclstm.multiwoz.loader.dataset_woz import SimpleDatasetWoz
from convlab2.nlg.sclstm.model.lm_deep import LMDeep
from convlab2.nlg.nlg import NLG

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "nlg-sclstm-multiwoz.zip")


def parse(is_user):
    if is_user:
        args = {
            'model_path': 'sclstm_usr.pt',
            'n_layer': 1,
            'beam_size': 10
        }
    else:
        args = {
            'model_path': 'sclstm.pt',
            'n_layer': 1,
            'beam_size': 10
        }

    config = configparser.ConfigParser()
    if is_user:
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config_usr.cfg'))
    else:
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config.cfg'))
    config.set('DATA', 'dir', os.path.dirname(os.path.abspath(__file__)))

    return args, config


class SCLSTM(NLG):
    def __init__(self, 
                 archive_file=DEFAULT_ARCHIVE_FILE, 
                 use_cuda=False,
                 is_user=False,
                 model_file='https://convlab.blob.core.windows.net/convlab-2/nlg_sclstm_multiwoz.zip'):

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for SC-LSTM is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'resource')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        self.USE_CUDA = use_cuda
        self.args, self.config = parse(is_user)
        self.dataset = SimpleDatasetWoz(self.config)

        # get model hyper-parameters
        hidden_size = self.config.getint('MODEL', 'hidden_size')

        # get feat size
        d_size = self.dataset.do_size + self.dataset.da_size + self.dataset.sv_size  # len of 1-hot feat
        vocab_size = len(self.dataset.word2index)

        self.model = LMDeep('sclstm', vocab_size, vocab_size, hidden_size, d_size, n_layer=self.args['n_layer'],
                            use_cuda=use_cuda)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.args['model_path'])
        # print(model_path)
        assert os.path.isfile(model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.eval()
        if use_cuda:
            self.model.cuda()

    def generate_delex(self, meta):
        for k, v in meta.items():
            domain, intent = k.split('-')
            if intent == "Request":
                for pair in v:
                    if type(pair[1]) != str:
                        pair[1] = str(pair[1])
                    pair.insert(1, '?')
            else:
                counter = {}
                for pair in v:
                    if type(pair[1]) != str:
                        pair[1] = str(pair[1])
                    if pair[0] == 'none':
                        pair.insert(1, 'none')
                    else:
                        if pair[0] in counter:
                            counter[pair[0]] += 1
                        else:
                            counter[pair[0]] = 1
                        pair.insert(1, str(counter[pair[0]]))

        # remove invalid dialog act
        meta_ = deepcopy(meta)
        for k, v in meta.items():
            for triple in v:
                voc = 'd-a-s-v:' + k + '-' + triple[0] + '-' + triple[1]
                if voc not in self.dataset.cardinality:
                    meta_[k].remove(triple)
            if not meta_[k]:
                del (meta_[k])
        meta = meta_

        # mapping the inputs
        do_idx, da_idx, sv_idx, featStr = self.dataset.getFeatIdx(meta)
        do_cond = [1 if i in do_idx else 0 for i in range(self.dataset.do_size)]  # domain condition
        da_cond = [1 if i in da_idx else 0 for i in range(self.dataset.da_size)]  # dial act condition
        sv_cond = [1 if i in sv_idx else 0 for i in range(self.dataset.sv_size)]  # slot/value condition
        feats = [do_cond + da_cond + sv_cond]

        feats_var = torch.FloatTensor(feats)
        if self.USE_CUDA:
            feats_var = feats_var.cuda()

        decoded_words = self.model.generate(self.dataset, feats_var, self.args['beam_size'])
        delex = decoded_words[0]  # (beam_size)
        
        return delex

    def generate_slots(self, meta):
        meta = deepcopy(meta)
        
        delex = self.generate_delex(meta)
        # get all informable or requestable slots
        slots = []
        for sen in delex:
            slot = []
            counter = {}
            words = sen.split()
            for word in words:
                if word.startswith('slot-'):
                    placeholder = word[5:]
                    if placeholder not in counter:
                        counter[placeholder] = 1
                    else:
                        counter[placeholder] += 1
                    slot.append(placeholder+'-'+str(counter[placeholder]))
            slots.append(slot)
            
        # for i in range(self.args.beam_size):
        #     print(i, slots[i])
            
        return slots[0]
    
    def generate(self, meta):
        """
        dialog_acts = [[intent, domain, slot, value], ... ]]
        =>
        meta = {"Attraction-Inform": [["Choice","many"],["Area","centre of town"]],
                "Attraction-Select": [["Type","church"],["Type"," swimming"],["Type"," park"]]}
        """
        # add placeholder value
        action = {}
        for intent, domain, slot, value in meta:
            k = '-'.join([domain, intent])
            action.setdefault(k, [])
            action[k].append([slot, value])
        meta = action

        delex = self.generate_delex(meta)
        
        # replace the placeholder with entities
        recover = []
        for sen in delex:
            counter = {}
            words = sen.split()
            for word in words:
                if word.startswith('slot-'):
                    flag = True
                    _, domain, intent, slot_type = word.split('-')
                    da = domain.capitalize() + '-' + intent.capitalize()
                    if da in meta:
                        key = da + '-' + slot_type.capitalize()
                        for pair in meta[da]:
                            if (pair[0].lower() == slot_type) and (
                                    (key not in counter) or (counter[key] == int(pair[1]) - 1)):
                                sen = sen.replace(word, pair[2], 1)
                                counter[key] = int(pair[1])
                                flag = False
                                break
                    if flag:
                        sen = sen.replace(word, '', 1)
            recover.append(sen)
            break

        # print('meta', meta)
        # for i in range(self.args.beam_size):
        #     print(i, delex[i])
        #     print(i, recover[i])

        return recover[0]
