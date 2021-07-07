import os
import re
import zipfile
import json
import torch
from unidecode import unidecode
import spacy
from convlab2.util.file_util import cached_path, get_root_path
from convlab2.nlu.nlu import NLU
from convlab2.nlu.jointBERT.dataloader import Dataloader
from convlab2.nlu.jointBERT.jointBERT import JointBERT
from convlab2.nlu.jointBERT.multiwoz.postprocess import recover_intent
from convlab2.nlu.jointBERT.multiwoz.preprocess import preprocess
from spacy.symbols import ORTH, LEMMA, POS


class BERTNLU(NLU):
    def __init__(self, mode='all', config_file='multiwoz_all_context.json',
                 model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_all_context.zip'):
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        self.mode = mode
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/{}'.format(config_file))
        config = json.load(open(config_file))
        # print(config['DEVICE'])
        # DEVICE = config['DEVICE']
        DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))

        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        print('Load from', best_model_path)
        model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.use_context = config['model']['context']
        self.dataloader = dataloader
        self.nlp = spacy.load('en_core_web_sm')
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            print('download en_core_web_sm for spacy')
            from spacy.cli.download import download as spacy_download
            spacy_download("en_core_web_sm")
            spacy_model_module = __import__("en_core_web_sm")
            self.nlp = spacy_model_module.load()
        with open(os.path.join(get_root_path(), 'data/multiwoz/db/postcode.json'), 'r') as f:
            token_list = json.load(f)

        for token in token_list:
            token = token.strip()
            self.nlp.tokenizer.add_special_case(token, [{ORTH: token, LEMMA: token, POS: u'NOUN'}])
        print("BERTNLU loaded")

    def predict(self, utterance, context=list()):
        # Note: spacy cannot tokenize 'id' or 'Id' correctly.
        utterance = re.sub(r'\b(id|Id)\b', 'ID', utterance)
        # tokenization first, very important!
        ori_word_seq = [token.text for token in self.nlp(unidecode(utterance)) if token.text.strip()]
        # print(ori_word_seq)
        ori_tag_seq = ['O'] * len(ori_word_seq)
        if self.use_context:
            if len(context) > 0 and type(context[0]) is list and len(context[0]) > 1:
                context = [item[1] for item in context]
            context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
            context_seq = context_seq[:512]
        else:
            context_seq = self.dataloader.tokenizer.encode('[CLS]')
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = self.dataloader.bert_tokenize(ori_word_seq, ori_tag_seq)
        word_seq = word_seq[:510]
        tag_seq = tag_seq[:510]
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq), self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        slot_logits, intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                                        context_seq_tensor=context_seq_tensor,
                                                        context_mask_tensor=context_mask_tensor)
        das = recover_intent(self.dataloader, intent_logits[0], slot_logits[0], tag_mask_tensor[0],
                             batch_data[0][0], batch_data[0][-4])
        dialog_act = []
        for intent, slot, value in das:
            domain, intent = intent.split('-')
            dialog_act.append([intent, domain, slot, value])
        return dialog_act


if __name__ == '__main__':
    text = "How about rosa's bed and breakfast ? Their postcode is cb22ha."
    nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                  model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_all_context.zip')
    print(nlu.predict(text))
    # text = "I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant.I don't care about the Price of the restaurant."
    # print(nlu.predict(text))
