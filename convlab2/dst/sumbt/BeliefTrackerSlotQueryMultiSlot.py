import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

from transformers import BertModel
from transformers import BertPreTrainedModel


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        return self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, encoder_hidden_states=output_all_encoded_layers)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


class BeliefTracker(nn.Module):
    def __init__(self, args, num_labels, device="cuda"):
        super(BeliefTracker, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.device = device

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_model_name, cache_dir=args.bert_model_cache_dir)
        self.utterance_encoder.train()
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_model_name, cache_dir=args.bert_model_cache_dir)
        self.sv_encoder.train()
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)

        ### RNN Belief Tracker
        self.nbt = None
        if args.task_name.find("gru") != -1:
            self.nbt = nn.GRU(input_size=self.bert_output_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.rnn_num_layers,
                              dropout=self.hidden_dropout_prob,
                              batch_first=True)
            self.init_parameter(self.nbt)
        elif args.task_name.find("lstm") != -1:
            self.nbt = nn.LSTM(input_size=self.bert_output_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.rnn_num_layers,
                               dropout=self.hidden_dropout_prob,
                               batch_first=True)
            self.init_parameter(self.nbt)
        if not self.zero_init_rnn:
            self.rnn_init_linear = nn.Sequential(
                nn.Linear(self.bert_output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.hidden_dropout_prob)
            )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # default evaluation mode
        self.eval()

    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(self.device)
        slot_mask = slot_ids > 0

        hid_slot, _ = self.sv_encoder(slot_ids.view(-1, self.max_label_length),
                                      slot_type_ids.view(-1, self.max_label_length),
                                      slot_mask.view(-1, self.max_label_length),
                                      output_all_encoded_layers=False)
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(self.device)
            label_mask = label_id > 0
            hid_label, _ = self.sv_encoder(label_id.view(-1, self.max_label_length),
                                           label_type_ids.view(-1, self.max_label_length),
                                           label_mask.view(-1, self.max_label_length),
                                           output_all_encoded_layers=False)
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")

    def _make_aux_tensors(self, ids, len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for i in range(len.size(0)):
            for j in range(len.size(1)):
                if len[i, j, 0] == 0:  # padding
                    break
                elif len[i, j, 1] > 0:  # escape only text_a case
                    start = len[i, j, 0]
                    ending = len[i, j, 0] + len[i, j, 1]
                    token_type_ids[i, j, start:ending] = 1
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    def forward(self, input_ids, input_len, labels, n_gpu=1, target_slot=None):

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # dialog size
        ts = input_ids.size(1)  # turn size
        bs = ds * ts
        slot_dim = len(target_slot)

        # Utterance encoding
        token_type_ids, attention_mask = self._make_aux_tensors(input_ids, input_len)

        hidden, _ = self.utterance_encoder(input_ids.view(-1, self.max_seq_length),
                                           token_type_ids.view(-1, self.max_seq_length),
                                           attention_mask.view(-1, self.max_seq_length),
                                           output_all_encoded_layers=False)
        hidden = torch.mul(hidden, attention_mask.view(-1, self.max_seq_length, 1).expand(hidden.size()).float())
        hidden = hidden.repeat(slot_dim, 1, 1)  # [(slot_dim*ds*ts), bert_seq, hid_size]

        hid_slot = self.slot_lookup.weight[target_slot, :]  # Select target slot embedding
        hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [(slot_dim*ds*ts), bert_seq, hid_size]

        # Attended utterance vector
        hidden = self.attn(hid_slot, hidden, hidden,
                           mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1))
        hidden = hidden.squeeze()  # [slot_dim*ds*ts, bert_dim]
        hidden = hidden.view(slot_dim, ds, ts, -1).view(-1, ts, self.bert_output_dim)

        # NBT
        if self.zero_init_rnn:
            h = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim).to(
                self.device)  # [1, slot_dim*ds, hidden]
        else:
            h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn_num_layers, 1, 1)
            h = self.rnn_init_linear(h)

        if isinstance(self.nbt, nn.GRU):
            rnn_out, _ = self.nbt(hidden, h)  # [slot_dim*ds, turn, hidden]
        elif isinstance(self.nbt, nn.LSTM):
            c = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim).to(
                self.device)  # [1, slot_dim*ds, hidden]
            rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
        rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out)))

        hidden = rnn_out.view(slot_dim, ds, ts, -1)

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_slot_labels, -1)
            _hidden = hidden[s, :, :, :].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(ds * ts * num_slot_labels,
                                                                                            -1)
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
                loss_slot.append(_loss.item())
                loss += _loss

        if labels is None:
            return output, torch.cat(pred_slot, 2)

        # calculate joint accuracy
        pred_slot = torch.cat(pred_slot, 2)
        # print('pred slot:', pred_slot[0][0])
        # print('labels:', labels[0][0])
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = torch.sum(accuracy, 0).float() \
                   / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        acc = sum(torch.sum(accuracy, 1) / slot_dim).float() \
              / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()  # joint accuracy

        if n_gpu == 1:
            return loss, loss_slot, acc, acc_slot, pred_slot
        else:
            return loss.unsqueeze(0), None, acc.unsqueeze(0), acc_slot.unsqueeze(0), pred_slot.unsqueeze(0)

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)
