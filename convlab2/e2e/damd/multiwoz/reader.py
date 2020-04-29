import numpy as np
import os, random, json, re
import spacy
from copy import deepcopy
from collections import Counter

from convlab2.e2e.damd.multiwoz.db_ops import MultiWozDB
from convlab2.e2e.damd.multiwoz.config import global_config as cfg
from convlab2.e2e.damd.multiwoz.clean_dataset import clean_text
from convlab2.e2e.damd.multiwoz.ontology import all_domains, get_slot, dialog_acts, dialog_act_params, eos_tokens
from convlab2.e2e.damd.multiwoz.utils import Vocab, padSeqs

DEFAULT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(DEFAULT_DIRECTORY, 'data/multi-woz-processed')

stopwords = ['and','are','as','at','be','been','but','by', 'for','however','if', 'not','of','on','or','so','the','there','was','were','whatever','whether','would']

class MultiWozReader(object):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')
        self.db = MultiWozDB(DEFAULT_DIRECTORY, cfg.dbs)
        self.vocab_size = self._build_vocab()
        # self.domain_files = json.loads(open(cfg.domain_file_path, 'r').read())
        self.slot_value_set = json.loads(open(os.path.join(DEFAULT_DIRECTORY, cfg.slot_value_set_path), 'r').read())

        self.delex_sg_valdict_path = os.path.join(default_path, 'delex_single_valdict.json')
        self.delex_mt_valdict_path = os.path.join(default_path, 'delex_multi_valdict.json')
        self.ambiguous_val_path = os.path.join(default_path, 'ambiguous_values.json')
        #self.delex_refs_path = os.path.join(default_path, 'reference_no.json')
        #self.delex_refs = json.loads(open(self.delex_refs_path, 'r').read())
        self.delex_sg_valdict = json.loads(open(self.delex_sg_valdict_path, 'r').read())
        self.delex_mt_valdict = json.loads(open(self.delex_mt_valdict_path, 'r').read())
        self.ambiguous_vals = json.loads(open(self.ambiguous_val_path, 'r').read())
        # if cfg.multi_acts_training:
        #     self.multi_acts = json.loads(open(cfg.multi_acts_path, 'r').read())

        self.reset()

    def reset(self):
        self.constraint_dict = {}
        self.turn_domain = ['general']
        self.py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_bsdx':None}
        self.book_state = {'train': False, 'restaurant': False, 'hotel':False}
        self.first_turn = True
        self.u = ''

    def _build_vocab(self):
        self.vocab = Vocab(cfg.vocab_size)
        self.vocab.load_vocab(os.path.join(DEFAULT_DIRECTORY, cfg.vocab_path_eval))
        return self.vocab.vocab_size


    def delex_by_valdict(self, text):
        text = clean_text(text)

        text = re.sub(r'\d{5}\s?\d{5,7}', '[value_phone]', text)
        text = re.sub(r'\d[\s-]stars?', '[value_stars]', text)
        text = re.sub(r'\$\d+|\$?\d+.?(\d+)?\s(pounds?|gbps?)', '[value_price]', text)
        text = re.sub(r'tr[\d]{4}', '[value_id]', text)
        text = re.sub(r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', '[value_postcode]', text)

        for value, slot in self.delex_mt_valdict.items():
            text = text.replace(value, '[value_%s]'%slot)

        for value, slot in self.delex_sg_valdict.items():
            tokens = text.split()
            for idx, tk in enumerate(tokens):
                if tk == value:
                    tokens[idx] = '[value_%s]'%slot
            text = ' '.join(tokens)

        for ambg_ent in self.ambiguous_vals:
            start_idx = text.find(' '+ambg_ent)   # ely is a place, but appears in words like moderately
            if start_idx == -1:
                continue
            front_words = text[:start_idx].split()
            ent_type = 'time' if ':' in ambg_ent else 'place'

            for fw in front_words[::-1]:
                if fw in ['arrive', 'arrives', 'arrived', 'arriving', 'arrival', 'destination', 'there', 'reach',  'to', 'by', 'before']:
                    slot = '[value_arrive]' if ent_type=='time' else '[value_destination]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)
                elif fw in ['leave', 'leaves', 'leaving', 'depart', 'departs', 'departing', 'departure',
                                'from', 'after', 'pulls']:
                    slot = '[value_leave]' if ent_type=='time' else '[value_departure]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)

        text = text.replace('[value_car] [value_car]', '[value_car]')
        return text

    def preprocess_utterance(self, user):
        u = ' '.join(clean_text(user).split())
        u_delex = self.delex_by_valdict(user)
        return u, u_delex

    def prepare_input_np(self, u, u_delex):

        self.u = u

        # get constraint dict

        inputs = {}
        py_batch = {}
        py_batch['user'] = [self.vocab.sentence_encode(u.split() + [eos_tokens['user']])]
        # py_batch['usdx'] = [self.vocab.sentence_encode(u_delex.split() + [eos_tokens['user']])]
        py_batch['usdx'] = py_batch['user']
        # py_batch['bspn'] = self.vocab.sentence_encode(constraints + [eos_tokens['bspn']])
        # py_batch['bsdx'] = [self.vocab.sentence_encode(cons_delex + [eos_tokens['bsdx']])]
        if self.first_turn:
            for item, py_list in self.py_prev.items():
                inputs[item+'_np'] = np.array([[1]])
                inputs[item+'_unk_np'] = np.array([[1]])
        else:
            for item, py_list in self.py_prev.items():
                if py_list is None:
                    continue
                if not cfg.enable_aspn and 'aspn' in item:
                    continue
                if not cfg.enable_bspn and 'bspn' in item:
                    continue
                prev_np = padSeqs(py_list, truncated=cfg.truncated, trunc_method='pre')
                inputs[item+'_np'] = prev_np
                if item in ['pv_resp', 'pv_bspn']:
                    inputs[item+'_unk_np'] = deepcopy(inputs[item+'_np'])
                    inputs[item+'_unk_np'][inputs[item+'_unk_np']>=self.vocab_size] = 2   # <unk>
                else:
                    inputs[item+'_unk_np'] = inputs[item+'_np']

        for item in ['user', 'usdx']:
            py_list = py_batch[item]
            trunc_method = 'post' if item == 'resp' else 'pre'
            inputs[item+'_np'] = padSeqs(py_list, truncated=cfg.truncated, trunc_method=trunc_method)
            if item in ['user', 'usdx']:
                inputs[item+'_unk_np'] = deepcopy(inputs[item+'_np'])
                inputs[item+'_unk_np'][inputs[item+'_unk_np']>=self.vocab_size] = 2   # <unk>
            else:
                inputs[item+'_unk_np'] = inputs[item+'_np']


        return inputs


    def bspan_to_constraint_dict(self, bspan, bspn_mode = 'bspn'):
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == eos_tokens[bspn_mode]:
                break
            if '[' in cons:
                if cons[1:-1] not in all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx+1]
                        ns = self.vocab.decode(ns) if type(ns) is not str else ns
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx+1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != eos_tokens[bspn_mode] and '[' not in vt and vt not in get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # print(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
        vector = self.db.addDBPointer(match_dom, match)
        return vector

    def aspan_to_act_list(self, aspan):
        aspan = aspan.split() if isinstance(aspan, str) else aspan
        acts = []
        domain = None
        conslen = len(aspan)
        for idx, cons in enumerate(aspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == eos_tokens['aspn']:
                break
            if '[' in cons and cons[1:-1] in dialog_acts:
                domain = cons[1:-1]

            elif '[' in cons and cons[1:-1] in dialog_act_params:
                if domain is None:
                    continue
                vidx = idx+1
                if vidx == conslen:
                    acts.append(domain+'-'+cons[1:-1]+'-none')
                    break
                vt = aspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                no_param_act = True
                while vidx < conslen and vt != eos_tokens['aspn'] and '[' not in vt:
                    no_param_act = False
                    acts.append(domain+'-'+cons[1:-1]+'-'+vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = aspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if no_param_act:
                    acts.append(domain+'-'+cons[1:-1]+'-none')

        return acts

    def dspan_to_domain(self, dspan):
        domains = {}
        dspan = dspan.split() if isinstance(dspan, str) else dspan
        for d in dspan:
            dom = self.vocab.decode(d) if type(d) is not str else d
            if dom != eos_tokens['dspn']:
                domains[dom] = 1
            else:
                break
        return domains

    def wrap_result(self, result_dict, eos_syntax=None):
        decode_fn = self.vocab.sentence_decode
        results = []
        eos_syntax = eos_tokens if not eos_syntax else eos_syntax

        if cfg.bspn_mode == 'bspn':
            field = ['dial_id', 'turn_num', 'user', 'bspn_gen','bspn', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'pointer']
        elif not cfg.enable_dst:
            field = ['dial_id', 'turn_num', 'user', 'bsdx_gen','bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'bspn', 'pointer']
        else:
            field = ['dial_id', 'turn_num', 'user', 'bsdx_gen','bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'bspn_gen','bspn', 'pointer']
        if self.multi_acts_record is not None:
            field.insert(7, 'multi_act_gen')

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
            for prop in field[2:]:
                entry[prop] = ''
            results.append(entry)
            for turn_no, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)
                    entry[key] = decode_fn(v, eos=eos_syntax[key]) if key in eos_syntax and v != '' else v
                results.append(entry)
        return results, field

    def restore(self, resp, domain, constraint_dict):
        restored = resp
        restored = restored.capitalize()
        restored = restored.replace(' -s', 's')
        restored = restored.replace(' -ly', 'ly')
        restored = restored.replace(' -er', 'er')


        mat_ents = self.db.get_match_num(constraint_dict, True)

        #ref =  random.choice(self.delex_refs)
        #restored = restored.replace('[value_reference]', ref.upper())
        restored = restored.replace('[value_car]', 'BMW')

        # restored.replace('[value_phone]', '830-430-6666')
        for d in domain:
            constraint = constraint_dict.get(d,None)
            if constraint:
                if 'stay' in constraint:
                    restored = restored.replace('[value_stay]', constraint['stay'])
                if 'day' in constraint:
                    restored = restored.replace('[value_day]', constraint['day'])
                if 'people' in constraint:
                    restored = restored.replace('[value_people]', constraint['people'])
                if 'time' in constraint:
                    restored = restored.replace('[value_time]', constraint['time'])
                if 'type' in constraint:
                    restored = restored.replace('[value_type]', constraint['type'])
                if d in mat_ents and len(mat_ents[d])==0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace('[value_price]', constraint['pricerange'])
                        if s+']' in restored:
                            restored = restored.replace('[value_%s]'%s, constraint[s])

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', str(random.choice([1,2,3,4,5])))


        # restored.replace('[value_car]', 'BMW')


        ent = mat_ents.get(domain[-1], [])
        if ent:
            # handle multiple [value_xxx] tokens first
            restored_split = restored.split()
            token_count = Counter(restored_split)
            for idx, t in enumerate(restored_split):
                if '[value' in t and token_count[t]>1 and token_count[t]<=len(ent):
                    slot = t[7:-1]
                    pattern = r'\['+t[1:-1]+r'\]'
                    for e in ent:
                        if e.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            if slot in ['name', 'address']:
                                rep = ' '.join([i.capitalize() if i not in stopwords else i for i in e[slot].split()])
                            elif slot in ['id','postcode']:
                                rep = e[slot].upper()
                            else:
                                rep = e[slot]
                            restored = re.sub(pattern, rep, restored, 1)
                        elif slot == 'price' and  e.get('pricerange'):
                            restored = re.sub(pattern, e['pricerange'], restored, 1)

            # handle normal 1 entity case
            ent = ent[0]
            for t in restored.split():
                if '[value' in t:
                    slot = t[7:-1]
                    if ent.get(slot):
                        if domain[-1] == 'hotel' and slot == 'price':
                            slot = 'pricerange'
                        if slot in ['name', 'address']:
                            rep = ' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                        elif slot in ['id','postcode']:
                            rep = ent[slot].upper()
                        else:
                            rep = ent[slot]
                        # rep = ent[slot]
                        restored = restored.replace(t, rep)
                        # restored = restored.replace(t, ent[slot])
                    elif slot == 'price' and  ent.get('pricerange'):
                        restored = restored.replace(t, ent['pricerange'])
                        # else:
                        #     print(restored, domain)


        restored = restored.replace('[value_phone]', '01223462354')
        restored = restored.replace('[value_postcode]', 'CB12DP')
        restored = restored.replace('[value_address]', 'Parkside, Cambridge')

        for t in restored.split():
            if '[value' in t:
                restored = restored.replace(t, 'UNKNOWN')

        restored = restored.split()
        for idx, w in enumerate(restored):
            if idx>0 and restored[idx-1] in ['.', '?', '!']:
                restored[idx]= restored[idx].capitalize()
        restored = ' '.join(restored)


        # if '[value_' in restored:

        #     print(domain)
        #     # print(mat_ents)
        #     print(resp)
        #     print(restored)
        return restored


if __name__=='__main__':
    reader = MultiWozReader()
    # for aspan in ["[general] [bye] [welcome] <eos_a>","[train] [inform] trainid destination arrive leave [offerbook] [general] [reqmore] <eos_a>",]:
    #     act = reader.aspan_to_constraint_dict(aspan.split())
    #     print(act)

    for bspan in ["[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday", "[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday <eos_b>"]:
        encoded=reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bspn')
        print(cons)
    for bspan in  ["[taxi] destination departure leave [hotel] name [attraction] name people day", "[taxi] destination departure leave [hotel] name [attraction] name people day <eos_b>"]:
        encoded=reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bsdx')
        print(cons)

