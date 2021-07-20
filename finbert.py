# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:13:31 2021

@author: MJH
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification



class FinBERT(pl.LightningModule):
    
    def __init__(self, model_path = 'model', train_samples = 3388, batch_size = 64, epochs = 10, num_labels = 3, learning_rate = 2e-5, discriminative_fine_tuning_rate = 0.85):
        super().__init__()
    
        self.learning_rate = learning_rate
        self.discriminative_fine_tuning_rate = discriminative_fine_tuning_rate
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.gradient_accumulation_steps = 1
        self.epochs = epochs
        self.warm_up_proportion = 0.2
        self.num_train_optimization_steps = int(self.train_samples / self.batch_size / self.gradient_accumulation_steps) * epochs
        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.warm_up_proportion)


        self.no_decay_layer_list = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
        config.num_labels = num_labels
        self.bert_model = BertForSequenceClassification.from_pretrained(model_path, config = config)
        
        self.optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
        
        self.criterion = nn.CrossEntropyLoss()
        
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels = None):        
        output = self.bert_model(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = labels
            )
         
        return output.loss, output.logits
    
    
    def get_optimizer_grouped_parameters(self):
        
        discriminative_fine_tuning_encoders = []
        for i in range(12):
            ith_layer = list(self.bert_model.bert.encoder.layer[i].named_parameters())
            
            encoder_decay = {
                'params': [param for name, param in ith_layer if
                           not any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
                'weight_decay': 0.01,
                'lr': self.learning_rate / (self.discriminative_fine_tuning_rate ** (12 - i))
                }
        
            encoder_nodecay = {
                'params': [param for name, param in ith_layer if
                           any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
                'weight_decay': 0.0,
                'lr': self.learning_rate / (self.discriminative_fine_tuning_rate ** (12 - i))}
            
            discriminative_fine_tuning_encoders.append(encoder_decay)
            discriminative_fine_tuning_encoders.append(encoder_nodecay)
            
        
        embedding_layer = self.bert_model.bert.embeddings.named_parameters()
        pooler_layer = self.bert_model.bert.pooler.named_parameters()
        classifier_layer = self.bert_model.classifier.named_parameters()
        
        optimizer_grouped_parameters = [
            {'params': [param for name, param in embedding_layer if
                        not any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.01,
             'lr': self.learning_rate / (self.discriminative_fine_tuning_rate ** 13)},
            {'params': [param for name, param in embedding_layer if
                        any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.0,
             'lr': self.learning_rate / (self.discriminative_fine_tuning_rate ** 13)},
            {'params': [param for name, param in pooler_layer if
                        not any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.01,
             'lr': self.learning_rate},
            {'params': [param for name, param in pooler_layer if
                        any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.0,
             'lr': self.learning_rate},
            {'params': [param for name, param in classifier_layer if
                        not any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.01,
             'lr': self.learning_rate},
            {'params': [param for name, param in classifier_layer if
                        any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.0,
             'lr': self.learning_rate}            
            ]
                
        optimizer_grouped_parameters.extend(discriminative_fine_tuning_encoders)
        
        return optimizer_grouped_parameters
    
    
    def training_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label = batch['label']
        
        loss, logits = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = label
            )
        
        total = label.size(0)        
        pred = torch.argmax(logits, 1).unsqueeze(1)
        correct = (pred == label).sum().item()
        acc = correct/total

        
        self.log('train_loss', loss, prog_bar = True, logger = True)
        self.log('train_acc', acc, prog_bar = True, logger = True)
        
        return loss
    
    
    def validation_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label = batch['label']

        
        loss, logits = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = label
            )
        
        total = label.size(0)        
        pred = torch.argmax(logits, 1).unsqueeze(1)
        correct = (pred == label).sum().item()
        acc = correct/total

        self.log('val_acc', acc, prog_bar = True, logger = True)
        self.log('val_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def test_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']        
        label = batch['label']

        loss, logits = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = label
            )
        
        total = label.size(0)        
        pred = torch.argmax(logits, 1).unsqueeze(1)
        correct = (pred == label).sum().item()
        acc = correct/total
        
        self.log('test_acc', acc, prog_bar = True, logger = True)
        self.log('test_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def configure_optimizers(self):
        
        optimizer = AdamW(
            self.optimizer_grouped_parameters,
            lr = self.learning_rate,
            correct_bias = False
            )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = self.num_warmup_steps,
            num_training_steps = self.num_train_optimization_steps
            )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]