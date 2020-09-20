import torch
from torch import nn
from transformers import BertModel
import numpy as np


class JointBERT(nn.Module):
    def __init__(self, model_config, device, slot_dim, intent_dim, req_dim, dataloader, intent_weight=None,req_weight=None):
        super(JointBERT, self).__init__()

        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.slot_intent_dim = dataloader.slot_intent_dim
        
        self.req_num_labels = req_dim
        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)
        self.req_weight = req_weight if req_weight is not None else torch.tensor([1.]*req_dim)

        print(model_config['pretrained_weights'])
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:
            if self.context:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.req_classifier = nn.Linear(self.hidden_units, self.req_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)  #linear input (N,*, input_size)  output(N, *,output_size)  only work for last layer
                self.intent_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
                self.req_hidden = nn.Linear(2*self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
            else:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.req_classifier = nn.Linear(self.hidden_units, self.req_num_labels)
                self.intent_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.req_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
            nn.init.xavier_uniform_(self.req_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.slot_num_labels)
                self.req_classifier = nn.linear(2*self.bert.config.hidden_size, self.req_num_labels)
            else:
                self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
                self.req_classifier = nn.Linear(self.bert.config.hidden_size, self.req_num_labels)
                self.slot_classifier = nn.Linear(self.bert.config.hidden_size, self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)
        nn.init.xavier_uniform_(self.req_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.req_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.req_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()


        self.intent_prefix = []
        for i in range(dataloader.slot_intent_dim):
            intent = dataloader.id2slotintent[i]
            #intent = self.dataloader.id2intent[i]
            domain, intent = intent.split('-')
            domain_id = dataloader.tokenizer.convert_tokens_to_ids(domain)#self.dataloader.tokenzier.convert_tokens_to_ids(domain)
            intent_id = dataloader.tokenizer.convert_tokens_to_ids(intent)
            self.intent_prefix.append([domain_id, intent_id])
        self.intent_prefix = torch.LongTensor(self.intent_prefix)
    
    def slot_forward(self,intent_logits,word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,context_seq_tensor=None, context_mask_tensor=None):
        batch_size = intent_logits.shape[0]
        intent_prefix = self.intent_prefix.repeat(batch_size,1).to(self.device)
        mask_prefix = torch.zeros(intent_prefix.shape,dtype=torch.long).to(self.device)

        max_len = word_seq_tensor.shape[1]
        repeat_word_seq_tensor = word_seq_tensor.unsqueeze(1).repeat(1,self.slot_intent_dim,1).view(-1,max_len)
        repeat_word_mask_tensor = word_mask_tensor.unsqueeze(1).repeat(1,self.slot_intent_dim,1).view(-1,max_len)
        slot_word_seq_tensor = torch.cat((repeat_word_seq_tensor[:,0].unsqueeze(1),intent_prefix,repeat_word_seq_tensor[:,1:]),1)
        slot_word_mask_tensor = torch.cat((mask_prefix,repeat_word_mask_tensor),1)

        
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids = slot_word_seq_tensor, 
                                    attention_mask = slot_word_mask_tensor)
        else:
            outputs = self.bert(input_ids = slot_word_seq_tensor, 
                                attention_mask = slot_word_mask_tensor)
        
        sequence_output = outputs[0]
        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(input_ids=context_seq_tensor,attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            context_max_len = context_output.shape[-1]
            context_output = context_output.unsqueeze(1).repeat(1,self.slot_intent_dim,1).view(-1,context_max_len)
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
        
        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
        
        sequence_output = self.dropout(sequence_output) 
        slot_logits = self.slot_classifier(sequence_output)
        output = (slot_logits,)

        if tag_seq_tensor is not None: 
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            if True in active_tag_loss:#deal with batch that totally masked
                active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
                active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]        
                slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels) / self.slot_intent_dim

            else:
                slot_loss = torch.tensor(0.0).to(self.device)
            output = output + (slot_loss,)

        return output #slot_logits, slot_loss

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None, req_tensor=None, context_seq_tensor=None, context_mask_tensor=None):
        
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0] 
        pooled_output = outputs[1]


        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
            
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)


        if self.hidden_units > 0:
            #sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
            intent_pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))
            req_pooled_output = nn.functional.relu(self.req_hidden(self.dropout(pooled_output)))

        '''
        sequence_output = self.dropout(sequence_output) #sequence used for slot classification [batchsize, sequence_length, hidden_size]
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)
        '''
        outputs = ()

        intent_pooled_output = self.dropout(intent_pooled_output) 
        intent_logits = self.intent_classifier(intent_pooled_output)

        slot_output = self.slot_forward(intent_logits, word_seq_tensor,word_mask_tensor, tag_seq_tensor,tag_mask_tensor,context_seq_tensor, context_mask_tensor)
        outputs = outputs + (slot_output[0],)#slot_logits

        req_pooled_output = self.dropout(req_pooled_output)
        req_logits = self.req_classifier(req_pooled_output)
        outputs = outputs + (intent_logits,)+(req_logits,)

        '''
        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss,)
        '''

        if tag_seq_tensor is not None:
            outputs = outputs + (slot_output[1],)#slot_loss
        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)
        
        if req_tensor is not None:
            req_loss = self.req_loss_fct(req_logits, req_tensor)
            outputs = outputs + (req_loss,)

        return outputs  # slot_logits, intent_logits,req_logits, (slot_loss), (intent_loss),(req_loss)
