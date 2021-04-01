import copy
from pprint import pprint
import random
from itertools import chain
import numpy as np
import zipfile

from matplotlib import pyplot as plt

from tensorboardX.writer import SummaryWriter
from tqdm._tqdm import trange, tqdm

from convlab2.util.file_util import cached_path

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup, AdamW

from convlab2.dst.dst import DST
from convlab2.dst.sumbt.crosswoz_en.convert_to_glue_format import convert_to_glue_format, trans_value
from convlab2.util.crosswoz_en.state import default_state

from convlab2.dst.sumbt.BeliefTrackerSlotQueryMultiSlot import BeliefTracker
from convlab2.dst.sumbt.crosswoz_en.sumbt_utils import *
from convlab2.dst.sumbt.crosswoz_en.sumbt_config import *

from convlab2.dst.sumbt.crosswoz_en.convert_to_glue_format import null

USE_CUDA = torch.cuda.is_available()
N_GPU = torch.cuda.device_count() if USE_CUDA else 1
DEVICE = "cuda" if USE_CUDA else "cpu"
ROOT_PATH = convlab2.get_root_path()
SUMBT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data/crosswoz_en')
DOWNLOAD_DIRECTORY = os.path.join(SUMBT_PATH, "pre-trained/")
crosswoz_en_slot_list = ['Attraction-duration', 'Attraction-fee', 'Attraction-name', 'Attraction-nearby attract.', 'Attraction-nearby hotels', 'Attraction-nearby rest.', 'Attraction-rating', 'Hotel-Hotel Facilities - 24-hour Hot Water', 'Hotel-Hotel Facilities - Bar', 'Hotel-Hotel Facilities - Breakfast Service', 'Hotel-Hotel Facilities - Broadband Internet', 'Hotel-Hotel Facilities - Business Center', 'Hotel-Hotel Facilities - Car Rental', 'Hotel-Hotel Facilities - Chess-Poker Room', 'Hotel-Hotel Facilities - Childcare Services', 'Hotel-Hotel Facilities - Chinese Restaurant', 'Hotel-Hotel Facilities - Disabled Facilities', 'Hotel-Hotel Facilities - Foreign Guests Reception', 'Hotel-Hotel Facilities - Free Breakfast Service', 'Hotel-Hotel Facilities - Free Domestic Long Distance Call', 'Hotel-Hotel Facilities - Free Local Calls', 'Hotel-Hotel Facilities - Gym', 'Hotel-Hotel Facilities - Hair Dryer', 'Hotel-Hotel Facilities - Heating', 'Hotel-Hotel Facilities - Hot Spring', 'Hotel-Hotel Facilities - Indoor Swimming Pool', 'Hotel-Hotel Facilities - International Call', 'Hotel-Hotel Facilities - Laundry Service', 'Hotel-Hotel Facilities - Luggage Storage', 'Hotel-Hotel Facilities - Meeting Room', 'Hotel-Hotel Facilities - Non-smoking Room', 'Hotel-Hotel Facilities - Outdoor Swimming Pool', 'Hotel-Hotel Facilities - Pay Parking', 'Hotel-Hotel Facilities - Pick-up Service', 'Hotel-Hotel Facilities - SPA', 'Hotel-Hotel Facilities - Sauna', 'Hotel-Hotel Facilities - Wake Up Service', 'Hotel-Hotel Facilities - Western Restaurant', 'Hotel-Hotel Facilities - WiFi in All Rooms', 'Hotel-Hotel Facilities - WiFi in Public Areas', 'Hotel-Hotel Facilities - WiFi in Public Areas and Some Rooms', 'Hotel-Hotel Facilities - WiFi in Some Rooms', 'Hotel-Hotel Facilities - WiFi throughout the Hotel', 'Hotel-name', 'Hotel-nearby attract.', 'Hotel-nearby hotels', 'Hotel-nearby rest.', 'Hotel-price', 'Hotel-rating', 'Hotel-type', 'Metro-from', 'Metro-to', 'Restaurant-cost', 'Restaurant-dishes', 'Restaurant-name', 'Restaurant-nearby attract.', 'Restaurant-nearby hotels', 'Restaurant-nearby rest.', 'Restaurant-rating', 'Taxi-from', 'Taxi-to']

def plot(x, y):
    a, b = [], []
    for x, y in sorted(zip(x, y)):
        a.append(x)
        b.append(y)
    plt.plot(a, b)

# def get_label_embedding(labels, max_seq_length, tokenizer, device):
#     features = []
#     for label in labels:
#         label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
#         label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
#         label_len = len(label_token_ids)

#         label_padding = [0] * (max_seq_length - len(label_token_ids))
#         label_token_ids += label_padding
#         assert len(label_token_ids) == max_seq_length

#         features.append((label_token_ids, label_len))

#     all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
#     all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

#     return all_label_token_ids, all_label_len


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class SUMBTTracker(DST):
    """
    Transferable multi-domain dialogue state tracker, adopted from https://github.com/SKTBrain/SUMBT
    """

    @staticmethod
    def init_data():
        if not os.path.exists(os.path.join(DATA_PATH, 'train.json.zip')):
            with zipfile.ZipFile(os.path.join(DATA_PATH, 'mt.zip')) as f:
                f.extractall(DATA_PATH)

        for split in ['train', 'test', 'val']:
            with zipfile.ZipFile(os.path.join(DATA_PATH, f'{split}.json.zip'), 'w') as f:
                f.write(os.path.join(DATA_PATH, f'{split}.json'), f'{split}.json')

    def __init__(self, data_dir=DATA_PATH):

        DST.__init__(self)

        # if not os.path.exists(data_dir):
        #     if model_file == '':
        #         raise Exception(
        #             'Please provide remote model file path in config')
        #     resp = urllib.request.urlretrieve(model_file)[0]
        #     temp_file = tarfile.open(resp)
        #     temp_file.extractall('data')
        #     assert os.path.exists(data_dir)

        processor = Processor(args)
        self.processor = processor
        # values of each slot e.g. values_list
        label_list = processor.get_labels()
        num_labels = [len(labels) for labels in label_list]  # number of slot-values in each slot-type

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_name, cache_dir=args.bert_model_cache_dir)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.device = torch.device("cuda" if USE_CUDA else "cpu")

        self.sumbt_model = BeliefTracker(args, num_labels, self.device)
        if USE_CUDA and N_GPU > 1:
            self.sumbt_model = torch.nn.DataParallel(self.sumbt_model)
        if args.fp16:
            self.sumbt_model.half()
        self.sumbt_model.to(self.device)

        ## Get slot-value embeddings
        self.label_token_ids, self.label_len = [], []
        for labels in label_list:
            # encoding values
            token_ids, lens = get_label_embedding(labels, args.max_label_length, self.tokenizer, self.device)
            self.label_token_ids.append(token_ids)
            self.label_len.append(lens)
        self.label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
        self.label_map_inv = [{i: label for i, label in enumerate(labels)} for labels in label_list]
        self.label_list = label_list
        self.target_slot = processor.target_slot
        ## Get domain-slot-type embeddings
        self.slot_token_ids, self.slot_len = \
            get_label_embedding(processor.target_slot, args.max_label_length, self.tokenizer, self.device)

        self.args = args
        self.state = default_state()
        self.param_restored = False
        if USE_CUDA and N_GPU == 1:
            self.sumbt_model.initialize_slot_value_lookup(self.label_token_ids, self.slot_token_ids)
        elif USE_CUDA and N_GPU > 1:
            self.sumbt_model.module.initialize_slot_value_lookup(self.label_token_ids, self.slot_token_ids)

        self.cached_res = {}
        convert_to_glue_format(DATA_PATH, SUMBT_PATH)
        if not os.path.isdir(os.path.join(SUMBT_PATH, args.output_dir)):
            os.makedirs(os.path.join(SUMBT_PATH, args.output_dir))
        self.train_examples = processor.get_train_examples(os.path.join(SUMBT_PATH, args.tmp_data_dir), accumulation=False)
        self.dev_examples = processor.get_dev_examples(os.path.join(SUMBT_PATH, args.tmp_data_dir), accumulation=False)
        self.test_examples = processor.get_test_examples(os.path.join(SUMBT_PATH, args.tmp_data_dir), accumulation=False)

    def load_weights(self, model_path=None):
        if model_path is None:
            model_ckpt = os.path.join(SUMBT_PATH, 'pre-trained/pytorch_model.bin')
        else:
            model_ckpt = model_path
        model = self.sumbt_model
        # in the case that slot and values are different between the training and evaluation
        if not USE_CUDA:
            ptr_model = torch.load(model_ckpt, map_location=torch.device('cpu'))
        else:
            ptr_model = torch.load(model_ckpt)
            print('loading pretrained weights')

        if not USE_CUDA or N_GPU == 1:
            state = model.state_dict()
            state.update(ptr_model)
            model.load_state_dict(state)
        else:
            # print("Evaluate using only one device!")
            model.module.load_state_dict(ptr_model)

        if USE_CUDA:
            model.to("cuda")

    def init_session(self):
        self.state = default_state()
        if not self.param_restored:
            if os.path.isfile(os.path.join(DOWNLOAD_DIRECTORY, 'pytorch_model.bin')):
                print('loading weights from downloaded model')
                self.load_weights(model_path=os.path.join(DOWNLOAD_DIRECTORY, 'pytorch_model.bin'))
            elif os.path.isfile(os.path.join(SUMBT_PATH, args.output_dir, 'pytorch_model.bin')):
                print('loading weights from trained model')
                self.load_weights(model_path=os.path.join(SUMBT_PATH, args.output_dir, 'pytorch_model.bin'))
            else:
                raise ValueError('no available weights found.')
            self.param_restored = True

    def construct_query(self, context):
        '''Construct query from context'''
        ids = []
        lens = []
        context_len = len(context)
        if context[0][0] != 'sys':
            context = [['sys', '']] + context
        for i in range(0, context_len, 2):
            # utt_user = ''
            # utt_sys = ''
            # for evaluation
            utt_sys = context[i][1]
            utt_user = context[i + 1][1]

            tokens_user = [x if x != '#' else '[SEP]' for x in self.tokenizer.tokenize(utt_user)]
            tokens_sys = [x if x != '#' else '[SEP]' for x in self.tokenizer.tokenize(utt_sys)]

            _truncate_seq_pair(tokens_user, tokens_sys, self.args.max_seq_length - 3)
            tokens = ["[CLS]"] + tokens_user + ["[SEP]"] + tokens_sys + ["[SEP]"]
            input_len = [len(tokens_user) + 2, len(tokens_sys) + 1]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.args.max_seq_length - len(input_ids))
            input_ids += padding
            assert len(input_ids) == self.args.max_seq_length
            ids.append(input_ids)
            lens.append(input_len)

        return (ids, lens)


    def update(self, user_act=None):
        if not isinstance(user_act, str):
            raise Exception(
                'Expected user_act is str but found {}'.format(type(user_act))
            )
        prev_state = self.state

        actual_history = copy.deepcopy(prev_state['history'])

        # if actual_history[-1][0] == 'user':
        #     actual_history[-1][1] += user_act
        # else:
        #     actual_history.append(['user', user_act])
        query = self.construct_query(actual_history)
        pred_states = self.predict(query)

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        for domain_slot, value in pred_states:
            domain, slot = domain_slot.split('-', 1)
            value = trans_value(value)

            # print(domain, slot, value)

            if domain not in new_belief_state:
                raise Exception(
                    'Error: domain <{}> not in belief state'.format(domain))

            domain_dic = new_belief_state[domain]
            if slot in domain_dic:
                domain_dic[slot] = value
            else:
                with open('sumbt_tracker_unknown_slot.log', 'a+') as f:
                    f.write(
                        'unknown slot name <{}> with value <{}> of domain <{}>\nitem: {}\n\n'.format(slot, value, domain, state)
                    )

        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        self.state = new_state
        return self.state

    def predict(self, query):
        cache_query_key = ''.join(str(list(chain.from_iterable(query[0]))))
        if cache_query_key in self.cached_res.keys():
            return self.cached_res[cache_query_key]

        input_ids, input_len = query
        input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
        input_len = torch.tensor(input_len).to(self.device).unsqueeze(0)
        labels = None
        _, pred_slot = self.sumbt_model(input_ids, input_len, labels)
        pred_slot_t = pred_slot[0][-1].tolist()
        predict_belief = []
        for idx, i in enumerate(pred_slot_t):
            predict_belief.append((self.target_slot[idx], self.label_map_inv[idx][i]))
            # predict_belief.append('{}-{}'.format(self.target_slot[idx], self.label_map_inv[idx][i]))
        self.cached_res[cache_query_key] = predict_belief

        return predict_belief

    def train(self, load_model=False, model_path=None):
        if load_model:
            if model_path is not None:
                self.load_weights(model_path)
        ## Training utterances
        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            self.train_examples, self.label_list, args.max_seq_length, self.tokenizer, args.max_turn_length)

        print('all input ids size: ', all_input_ids.size())
        num_train_batches = all_input_ids.size(0)
        num_train_steps = int(
            num_train_batches / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** training *****")
        logger.info("  Num examples = %d", len(self.train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(DEVICE), all_input_len.to(
            DEVICE), all_label_ids.to(DEVICE)

        train_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        all_input_ids_dev, all_input_len_dev, all_label_ids_dev = convert_examples_to_features(
            self.dev_examples, self.label_list, args.max_seq_length, self.tokenizer, args.max_turn_length)

        logger.info("***** validation *****")
        logger.info("  Num examples = %d", len(self.dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)

        all_input_ids_dev, all_input_len_dev, all_label_ids_dev = \
            all_input_ids_dev.to(DEVICE), all_input_len_dev.to(DEVICE), all_label_ids_dev.to(DEVICE)

        dev_data = TensorDataset(all_input_ids_dev, all_input_len_dev, all_label_ids_dev)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

        logger.info("Loaded data!")

        if args.fp16:
            self.sumbt_model.half()
        self.sumbt_model.to(DEVICE)

        # ## Get domain-slot-type embeddings
        # slot_token_ids, slot_len = \
        #     get_label_embedding(self.processor.target_slot, args.max_label_length, self.tokenizer, DEVICE)

        # # for slot_idx, slot_str in zip(slot_token_ids, self.processor.target_slot):
        # #     self.idx2slot[slot_idx] = slot_str

        # ## Get slot-value embeddings
        # label_token_ids, label_len = [], []
        # for slot_idx, labels in zip(slot_token_ids, self.label_list):
        #     # self.idx2value[slot_idx] = {}
        #     token_ids, lens = get_label_embedding(labels, args.max_label_length, self.tokenizer, DEVICE)
        #     label_token_ids.append(token_ids)
        #     label_len.append(lens)
        #     # for label, token_id in zip(labels, token_ids):
        #     #     self.idx2value[slot_idx][token_id] = label

        # logger.info('embeddings prepared')

        # if USE_CUDA and N_GPU > 1:
        #     self.sumbt_model.module.initialize_slot_value_lookup(label_token_ids, slot_token_ids)
        # else:
        #     self.sumbt_model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)

        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.learning_rate},
            ]
            return optimizer_grouped_parameters

        if not USE_CUDA or N_GPU == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.sumbt_model)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.sumbt_model.module)

        t_total = num_train_steps

        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.fp16_loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.fp16_loss_scale)

        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_proportion*t_total, num_training_steps=t_total)
        logger.info(optimizer)

        # Training code
        ###############################################################################

        print(torch.cuda.memory_allocated())

        logger.info("Training...")

        global_step = 0
        last_update = None
        best_loss = None
        model = self.sumbt_model
        if not args.do_not_use_tensorboard:
            summary_writer = None
        else:
            summary_writer = SummaryWriter("./tensorboard_summary/logs_1214/")

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, input_len, label_ids = batch
                # print(input_ids.size())

                # Forward
                if N_GPU == 1:
                    loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, N_GPU)
                else:
                    loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, N_GPU)

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # tensrboard logging
                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch", epoch, global_step)
                    summary_writer.add_scalar("Train/Loss", loss, global_step)
                    summary_writer.add_scalar("Train/JointAcc", acc, global_step)
                    if N_GPU == 1:
                        for i, slot in enumerate(self.processor.target_slot):
                            summary_writer.add_scalar("Train/Loss_%s" % slot.replace(' ', '_'), loss_slot[i],
                                                      global_step)
                            summary_writer.add_scalar("Train/Acc_%s" % slot.replace(' ', '_'), acc_slot[i], global_step)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify lealrning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    if summary_writer is not None:
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    if scheduler is not None:
                        torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters, 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1


            # Perform evaluation on validation dataset
            model.eval()
            dev_loss = 0
            dev_acc = 0
            dev_loss_slot, dev_acc_slot = None, None
            nb_dev_examples, nb_dev_steps = 0, 0

            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, input_len, label_ids = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)

                with torch.no_grad():
                    if N_GPU == 1:
                        loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, N_GPU)
                    else:
                        loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, N_GPU)

                        # average to multi-gpus
                        loss = loss.mean()
                        acc = acc.mean()
                        acc_slot = acc_slot.mean(0)

                num_valid_turn = torch.sum(label_ids[:, :, 0].view(-1) > -1, 0).item()
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn

                if N_GPU == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn

                nb_dev_examples += num_valid_turn


            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples

            if N_GPU == 1:
                dev_acc_slot = dev_acc_slot / nb_dev_examples

            # tensorboard logging
            if summary_writer is not None:
                summary_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Validate/Acc", dev_acc, global_step)
                if N_GPU == 1:
                    for i, slot in enumerate(self.processor.target_slot):
                        summary_writer.add_scalar("Validate/Loss_%s" % slot.replace(' ', '_'),
                                                  dev_loss_slot[i] / nb_dev_examples, global_step)
                        summary_writer.add_scalar("Validate/Acc_%s" % slot.replace(' ', '_'), dev_acc_slot[i],
                                                  global_step)

            dev_loss = round(dev_loss, 6)

            output_model_file = os.path.join(os.path.join(SUMBT_PATH, args.output_dir), "pytorch_model.bin")

            if last_update is None or dev_loss < best_loss:
                last_update = epoch
                best_loss = dev_loss
                best_acc = dev_acc
                if not USE_CUDA or N_GPU == 1:
                    torch.save(model.state_dict(), output_model_file)
                else:
                    torch.save(model.module.state_dict(), output_model_file)

                logger.info(
                    "*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f, global_step=%d ***" % (
                        last_update, best_loss, best_acc, global_step))
            else:
                logger.info(
                    "*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f, global_step=%d  ***" % (
                        epoch, dev_loss, dev_acc, global_step))

            if last_update + args.patience <= epoch:
                break

    def test(self, mode='dev', model_path=os.path.join(os.path.join(SUMBT_PATH, args.output_dir), "pytorch_model.bin")):
        '''Testing funciton of TRADE (to be added)'''
        # Evaluation
        self.load_weights(model_path)

        if mode == 'test':
            eval_examples = self.dev_examples
        elif mode == 'dev':
            eval_examples = self.test_examples

        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            eval_examples, self.label_list, args.max_seq_length, self.tokenizer, args.max_turn_length)
        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(DEVICE), all_input_len.to(
            DEVICE), all_label_ids.to(DEVICE)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)

        eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.dev_batch_size)

        model = self.sumbt_model
        eval_loss, eval_accuracy = 0, 0
        eval_loss_slot, eval_acc_slot = None, None
        nb_eval_steps, nb_eval_examples = 0, 0

        accuracies = {'joint7': 0, 'slot7': 0, 'joint5': 0, 'slot5': 0, 'joint_rest': 0, 'slot_rest': 0,
                      'num_turn': 0, 'num_slot7': 0, 'num_slot5': 0, 'num_slot_rest': 0}

        for input_ids, input_len, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            # if input_ids.dim() == 2:
            #     input_ids = input_ids.unsqueeze(0)
            #     input_len = input_len.unsqueeze(0)
            #     label_ids = label_ids.unsuqeeze(0)

            with torch.no_grad():
                if not USE_CUDA or N_GPU == 1:
                    loss, loss_slot, acc, acc_slot, pred_slot = model(input_ids, input_len, label_ids, 1)
                else:
                    loss, _, acc, acc_slot, pred_slot = model(input_ids, input_len, label_ids, N_GPU)
                    nbatch = label_ids.size(0)
                    nslot = pred_slot.size(3)
                    pred_slot = pred_slot.view(nbatch, -1, nslot)

            accuracies = eval_all_accs(pred_slot, label_ids, accuracies)

            nb_eval_ex = (label_ids[:, :, 0].view(-1) != -1).sum().item()
            nb_eval_examples += nb_eval_ex
            nb_eval_steps += 1

            if not USE_CUDA or N_GPU == 1:
                eval_loss += loss.item() * nb_eval_ex
                eval_accuracy += acc.item() * nb_eval_ex
                if eval_loss_slot is None:
                    eval_loss_slot = [l * nb_eval_ex for l in loss_slot]
                    eval_acc_slot = acc_slot * nb_eval_ex
                else:
                    for i, l in enumerate(loss_slot):
                        eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
                    eval_acc_slot += acc_slot * nb_eval_ex
            else:
                eval_loss += sum(loss) * nb_eval_ex
                eval_accuracy += sum(acc) * nb_eval_ex

            # exit(1)

        eval_loss = eval_loss / nb_eval_examples
        eval_accuracy = eval_accuracy / nb_eval_examples
        if not USE_CUDA or N_GPU == 1:
            eval_acc_slot = eval_acc_slot / nb_eval_examples

        loss = None

        if not USE_CUDA or N_GPU == 1:
            result = {
                # 'num': '\t'.join([str(x) for x in model.num_labels]),
                'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy,
                'loss': loss,
                'eval_loss_slot': '\t'.join([str(val / nb_eval_examples) for val in eval_loss_slot]),
                'eval_acc_slot': '\t'.join([str((val).item()) for val in eval_acc_slot]),
            }
        else:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'loss': loss
                      }

        out_file_name = 'eval_results'
        # if TARGET_SLOT == 'all':
        #     out_file_name += '_all'
        output_eval_file = os.path.join(os.path.join(SUMBT_PATH, args.output_dir), "%s.txt" % out_file_name)

        if not USE_CUDA or N_GPU == 1:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        out_file_name = 'eval_all_accuracies'
        with open(os.path.join(os.path.join(SUMBT_PATH, args.output_dir), "%s.txt" % out_file_name), 'w') as f:
            s = '{:^22s}:{:^22s}:{:^22s}:{:^22s}:{:^22s}:{:^22s}'.format(
                'joint acc (7 domain)', 
                'slot acc (7 domain)', 
                'joint acc (5 domain)', 
                'slot acc (5 domain)', 
                'joint restaurant', 
                'slot acc restaurant')
            f.write(s + '\n')
            print(s)
            s = '{:^22.5f}:{:^22.5f}:{:^22.5f}:{:^22.5f}:{:^22.5f}:{:^22.5f}'.format(
                (accuracies['joint7'] / accuracies['num_turn']).item(),
                (accuracies['slot7'] / accuracies['num_slot7']).item(),
                (accuracies['joint5'] / accuracies['num_turn']).item(),
                (accuracies['slot5'] / accuracies['num_slot5']).item(),
                (accuracies['joint_rest'] / accuracies['num_turn']).item(),
                (accuracies['slot_rest'] / accuracies['num_slot_rest']).item()
            )
            f.write(s + '\n')
            print(s)
