from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import shutil, zipfile
import pickle

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import json
# import pandas as pd
import copy

from convlab2.dst.trade.multiwoz.utils.masked_cross_entropy import *
from convlab2.dst.trade.multiwoz.utils.config import *
from convlab2.util.file_util import cached_path
from convlab2.dst.rule.multiwoz import normalize_value
from convlab2.dst.trade.multiwoz.utils.utils_multiWOZ_DST import get_slot_information, Lang, read_langs2, get_seq, read_langs
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA

from convlab2.dst.trade.multiwoz.utils.utils_multiWOZ_DST import prepare_data_seq
from convlab2.dst.trade.trade import TRADE
from convlab2.util.multiwoz.state import default_state


class MultiWOZTRADE(TRADE, nn.Module):
    def __init__(self, argpath=None, mode='en', training=False):
        super(MultiWOZTRADE, self).__init__()

        if argpath is None:
            argpath = 'model/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.3591'
        args['path'] = argpath

        self.init_session()

        self.multiwoz_root = os.path.dirname(os.path.abspath(__file__))
        self.download_model()
        self.download_data()

        directory = argpath.split("/")
        HDD = directory[2].split('HDD')[1].split('BSZ')[0]
        decoder = directory[1].split('-')[0]
        BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
        args["decoder"] = decoder

        def parent_dir(path, time=1):
            for _ in range(time):
                path = os.path.dirname(path)
            return path
        root_dir = parent_dir(os.path.abspath(__file__), 5)

        # load ontology variables
        self.value_dict = json.load(open(os.path.join(root_dir, 'data/multiwoz/value_dict.json')))
        self.det_dic = {}
        for domain, dic in REF_USR_DA.items():
            for key, value in dic.items():
                assert '-' not in key
                self.det_dic[key.lower()] = key + '-' + domain
                self.det_dic[value.lower()] = key + '-' + domain

        train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(False, 'dst',
                                                                                                      False,
                                                                                                      batch_size=4)
        self.slot_list = SLOTS_LIST
        self.test_set = test
        hidden_size = int(HDD)
        lang = lang
        path = argpath
        lr=0
        task = 'dst'
        dropout = 0
        slots = SLOTS_LIST
        gating_dict = gating_dict
        nb_train_vocab = max_word

        self.mode = mode
        self.name = "TRADE"
        self.task = task
        self.hidden_size = hidden_size
        self.lang = lang[0]
        self.mem_lang = lang[1]
        self.lr = lr
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()

        self.init_run_vars(training)

        self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout, mode=mode)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout,
                                 self.slots, self.nb_gate)
        model_root = os.path.dirname(os.path.abspath(__file__))
        if path:
            path = os.path.join(model_root, path)
            map_loc = 'cpu' if not torch.cuda.is_available() else None
            self.encoder.load_state_dict(torch.load(str(path) + '/enc.pr', map_location=map_loc))
            self.decoder.load_state_dict(torch.load(str(path) + '/dec.pr', map_location=map_loc))
            if torch.cuda.is_available():
                self.encoder = self.encoder.cuda()
                self.decoder = self.decoder.cuda()

        # Initialize optimizers and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def init_session(self):
        self.state = default_state()

    def download_model(self, model_url="https://convlab.blob.core.windows.net/convlab-2/trade_multiwoz_model.zip"):
        """Automatically download the pretrained model and necessary data."""
        if os.path.exists(os.path.join(self.multiwoz_root, 'model/TRADE-multiwozdst')):
            return
        model_dir = os.path.join(self.multiwoz_root, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        zip_file_path = os.path.join(model_dir, 'trade_multiwoz_model.zip')
        if not os.path.exists(os.path.join(model_dir, 'trade_multiwoz_model.zip')):
            print('downloading multiwoz Trade model files...')
            cached_path(model_url, model_dir)
            files = os.listdir(model_dir)
            target_file = ''
            for name in files:
                if name.endswith('.json'):
                    target_file = name[:-5]
            try:
                assert target_file in files
            except Exception as e:
                print('allennlp download file error: TRADE Cross model download failed.')
                raise e
            shutil.copyfile(os.path.join(model_dir, target_file), zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            print('unzipping model file ...')
            zip_ref.extractall(model_dir)

    def download_data(self, data_url="https://convlab.blob.core.windows.net/convlab-2/trade_multiwoz_data.zip"):
        """Automatically download the pretrained model and necessary data."""
        if os.path.exists(os.path.join(self.multiwoz_root, 'data/multi-woz')) and \
                os.path.exists(os.path.join(self.multiwoz_root, 'data/dev_dials.json')):
            return
        data_dir = os.path.join(self.multiwoz_root, 'data')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        zip_file_path = os.path.join(data_dir, 'trade_multiwoz_data.zip')
        if not os.path.exists(os.path.join(data_dir, 'trade_multiwoz_data.zip')):
            print('downloading multiwoz TRADE data files...')
            cached_path(data_url, data_dir)
            files = os.listdir(data_dir)
            target_file = ''
            for name in files:
                if name.endswith('.json'):
                    target_file = name[:-5]
            try:
                assert target_file in files
            except Exception as e:
                print('allennlp download file error: TRADE Cross model download failed.')
                raise e
            shutil.copyfile(os.path.join(data_dir, target_file), zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg, print_loss_ptr, print_loss_gate)

    def save_model(self, dec_type):
        if self.mode == 'en':
            directory = 'model/TRADE-' + args["addName"] + args['dataset'] + str(self.task) + '/' + 'HDD' + str(
                self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + str(dec_type)
        else:
            if data_version == 'init':
                directory = 'save_cn/TRADE-' + args["addName"] + args['dataset'] + str(self.task) + '/' + 'HDD' + str(
                    self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + str(dec_type)
            else:
                directory = 'save_cn_processed/TRADE-' + args["addName"] + args['dataset'] + str(
                    self.task) + '/' + 'HDD' + str(self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(
                    self.dropout) + str(dec_type)

        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, slot_temp, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]
        all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(data, use_teacher_forcing,
                                                                                            slot_temp)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(),  # [:,:len(self.point_slots)].contiguous(),
            data["y_lengths"])  # [:,:len(self.point_slots)])
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
                                       data["gating_label"].contiguous().view(-1))

        if args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr

        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, clip):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def optimize_GEM(self, clip):
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):
        # Build unknown mask for memory to encourage generalization
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            if USE_CUDA:
                rand_mask = rand_mask.cuda()
            story = data['context'] * rand_mask.long()
        else:
            story = data['context']

        if torch.cuda.is_available():
            story = story.cuda()
        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, \
                                                                                                     encoded_hidden,
                                                                                                     encoded_outputs,
                                                                                                     data[
                                                                                                         'context_len'],
                                                                                                     story, max_res_len,
                                                                                                     data['generate_y'], \
                                                                                                     use_teacher_forcing,
                                                                                                     slot_temp)
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def evaluate(self, early_stop=None):
        dev = self.test_set
        matric_best = 1e7
        slot_temp = self.slot_list[3]
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            # if j == 10: break  # test code
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, gates, words, class_words = self.encode_and_decode(data_dev, False, slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                    "turn_belief": data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                if args["use_gate"]:
                    for si, sg in enumerate(gate):
                        if sg == self.gating_dict["none"]:
                            continue
                        elif sg == self.gating_dict["ptr"]:
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e == 'EOS':
                                    break
                                else:
                                    st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS':
                                break
                            else:
                                st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                    print("True", set(data_dev["turn_belief"][bi]))
                    print("Pred", set(predict_belief_bsz_ptr), "\n")

        if args["genSample"]:
            json.dump(all_prediction, open("all_prediction_{}.json".format(self.name), 'w'), indent=4)

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr",
                                                                                      slot_temp)

        evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr,
                              "Joint F1": F1_score_ptr}
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        joint_acc_score = joint_acc_score_ptr  # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                print("MODEL SAVED")
            return joint_acc_score

    def detect_requestable_slots(self, observation):
        result = {}
        observation = observation.lower()
        _observation = ' {} '.format(observation)
        for value in self.det_dic.keys():
            _value = ' {} '.format(value.strip())
            if _value in _observation:
                key, domain = self.det_dic[value].split('-')
                if domain not in result:
                    result[domain] = {}
                result[domain][key] = 0
        return result

    def update(self, user_act):
        if type(user_act) is not str:
            raise Exception('Expected user_act to be <class \'str\'> type, but get {}.'.format(type(user_act)))
        prev_state = self.state

        batch_size = 1
        # reformat input sequence
        context = ' ; '.join([item[1].strip().lower() for item in self.state['history']]).strip()
        sentence = self.state['history'][-1][1].strip().lower()
        _, _, dev, _, _, SLOTS_LIST, _, _ = self.reformat_input(False,
                                                                  'dst',
                                                                  False,
                                                                  batch_size=batch_size,
                                                                  source_text=context,
                                                                  curr_utterance=sentence)
        slot_temp = SLOTS_LIST[3]


        # Set to not-training mode to disable dropout
        # self.encoder.train(False)
        # self.decoder.train(False)
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        for j, data_dev in enumerate(dev):
            assert j == 0
            # if MODE == 'cn' and j >= 450:
            #     break
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, gates, words, class_words = self.encode_and_decode(data_dev, False, slot_temp)
            for bi in range(1):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                    "turn_belief": data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                if args["use_gate"]:
                    for si, sg in enumerate(gate):
                        if sg == self.gating_dict["none"]:
                            continue
                        elif sg == self.gating_dict["ptr"]:
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e == 'EOS':
                                    break
                                else:
                                    st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS':
                                break
                            else:
                                st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                with open('./test_trade_log.txt', 'a+') as f:
                    f.write('{}\n'.format(prev_state['history'][-2:]))
                    f.write('{}\n\n'.format(predict_belief_bsz_ptr))
                new_belief_state = self.reformat_belief_state(predict_belief_bsz_ptr, copy.deepcopy(prev_state['belief_state']))
                self.state['belief_state'] = new_belief_state
                ## request state
                new_request_state = copy.deepcopy(self.state['request_state'])
                user_request_slot = self.detect_requestable_slots(user_act.lower())
                for domain in user_request_slot:
                    for key in user_request_slot[domain]:
                        if domain not in new_request_state:
                            new_request_state[domain] = {}
                        if key not in new_request_state[domain]:
                            new_request_state[domain][key] = user_request_slot[domain][key]
                self.state['request_state'] = new_request_state
                return self.state

    def init_run_vars(self, training=False):
        root_path = os.path.dirname(os.path.abspath(__file__))
        # Create saving folder
        if args['path']:
            folder_name = '/'.join(args['path'].rsplit('/', 2)[:2]) + '/'
        else:
            folder_name = 'model/{}-'.format(args["decoder"]) + args["addName"] + args['dataset'] + str(
                args['task']) + '/'
        folder_name = root_path + '/' + folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # load domain-slot pairs from ontology
        self.ontology = json.load(open(root_path + "/data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
        self.ALL_SLOTS = get_slot_information(self.ontology)
        self.gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
        # Vocabulary
        lang, mem_lang = Lang(), Lang()
        lang.index_words(self.ALL_SLOTS, 'slot')
        mem_lang.index_words(self.ALL_SLOTS, 'slot')
        self.lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
        self.mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'
        self.lang = lang
        self.mem_lang = mem_lang

        if not training:
            with open(folder_name + self.lang_name, 'rb') as handle:
                self.lang = pickle.load(handle)
            with open(folder_name + self.mem_lang_name, 'rb') as handle:
                self.mem_lang = pickle.load(handle)

    def reformat_input(self, training, task="dst", sequicity=0, batch_size=100, source_text='', curr_utterance=''):
        training = False
        eval_batch = 1  # args["eval_batch"] if args["eval_batch"] else batch_size
        root_path = os.path.dirname(os.path.abspath(__file__))


        if not training:

            pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
            utterance = curr_utterance
            if source_text == '':
                source_text = "; " + utterance.strip() + " ;"
            pair_test, test_max_len, slot_test = read_langs2(source_text, utterance, self.gating_dict, self.ALL_SLOTS, "test",
                                                             self.lang,
                                                             self.mem_lang, sequicity, training)
            test = get_seq(pair_test, self.lang, self.mem_lang, eval_batch, False, sequicity)

        test_4d = []
        if args['except_domain'] != "":
            root_path = os.path.dirname(os.path.abspath(__file__))
            file_test = root_path + '/data/test_dials.json'
            pair_test_4d, _, _ = read_langs(file_test, self.gating_dict, self.ALL_SLOTS, "dev", self.lang, self.mem_lang, sequicity,
                                            training)
            test_4d = get_seq(pair_test_4d, self.lang, self.mem_lang, eval_batch, False, sequicity)


        SLOTS_LIST = [self.ALL_SLOTS, [], [], slot_test]
        LANG = [self.lang, self.mem_lang]
        return None, None, test, test_4d, LANG, SLOTS_LIST, self.gating_dict, nb_train_vocab

    def reformat_belief_state(self, raw_state, bs):
        for item in raw_state:
            item = item.lower()
            slist = item.split('-', 2)
            domain = slist[0].strip()
            slot = slist[1].strip()
            value = slist[2].strip()
            if domain not in bs:
                raise Exception('Error: domain <{}> not in belief state'.format(domain))
            dbs = bs[domain]
            assert 'semi' in dbs
            assert 'book' in dbs
            slot = REF_SYS_DA[domain.capitalize()].get(slot, slot)
            # reformat some slots
            if slot == 'arriveby':
                slot = 'arriveBy'
            elif slot == 'leaveat':
                slot = 'leaveAt'
            if slot in dbs['semi']:
                dbs['semi'][slot] = normalize_value(self.value_dict, domain, slot, value)
            elif slot in dbs['book']:
                dbs['book'][slot] = value
            elif slot.lower() in dbs['book']:
                dbs['book'][slot.lower()] = value
            else:
                with open('trade_multiwoz_unknown_slot.log', 'a+') as f:
                    f.write('unknown slot name <{}> with value <{}> of domain <{}>\nitem: {}\n\n'.format(slot, value,
                            domain, item))
        return bs

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total != 0 else 0
        turn_acc_score = turn_acc / float(total) if total != 0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            if len(pred) == 0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1, mode='en'):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        if args["load_embedding"]:
            if mode == 'en':
                with open(os.path.join("data/", 'emb_en{}.json'.format(vocab_size))) as f:
                    E = json.load(f)
            else:
                with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                    E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches,
                use_teacher_forcing, slot_temp):
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        if USE_CUDA:
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()

        # Get the slot embedding
        slot_emb_dict = {}
        for i, slot in enumerate(slot_temp):
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if USE_CUDA: domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if USE_CUDA: slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        if args["parallel_decode"]:
            # Compute pointer-generator output, puting all (domain, slot) in one batch
            decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size)  # (batch*|slot|) * emb
            hidden = encoded_hidden.repeat(1, len(slot_temp), 1)  # 1 * (batch*|slot|) * emb
            words_point_out = [[] for i in range(len(slot_temp))]
            words_class_out = []

            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

                if wi == 0:
                    all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros(p_vocab.size())
                if USE_CUDA: p_context_ptr = p_context_ptr.cuda()

                p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words = [self.lang.index2word[w_idx.item()] for w_idx in pred_word]

                for si in range(len(slot_temp)):
                    words_point_out[si].append(words[si * batch_size:(si + 1) * batch_size])

                all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab,
                                                               (len(slot_temp), batch_size, self.vocab_size))

                if use_teacher_forcing:
                    decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1, 0)))
                else:
                    decoder_input = self.embedding(pred_word)

                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            # Compute pointer-generator output, decoding each (domain, slot) one-by-one
            words_point_out = []
            counter = 0
            for slot in slot_temp:
                hidden = encoded_hidden
                words = []
                slot_emb = slot_emb_dict[slot]
                decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
                for wi in range(max_res_len):
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                    context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                    if wi == 0:
                        all_gate_outputs[counter] = self.W_gate(context_vec)
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                    p_context_ptr = torch.zeros(p_vocab.size())
                    if USE_CUDA: p_context_ptr = p_context_ptr.cuda()
                    p_context_ptr.scatter_add_(1, story, prob)
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                    pred_word = torch.argmax(final_p_vocab, dim=1)
                    words.append([self.lang.index2word[w_idx.item()] for w_idx in pred_word])
                    all_point_outputs[counter, :, wi, :] = final_p_vocab
                    if use_teacher_forcing:
                        decoder_input = self.embedding(target_batches[:, counter, wi])  # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)
                    if USE_CUDA: decoder_input = decoder_input.cuda()
                counter += 1
                words_point_out.append(words)

        return all_point_outputs, all_gate_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
