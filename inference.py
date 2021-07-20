# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:23:11 2021

@author: MJH
"""

from finbert import FinBERT
from transformers import BertTokenizer
from auxiliary import categorizer, convert_idx_to_sentiment

import torch


def get_sentiment(text):
    
    encoded_text = tokenizer.encode_plus(
        text,
        max_length = 512,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = 'pt'
        )
    
    logit_output = sentiment_model(
        input_ids = encoded_text.input_ids.flatten().unsqueeze(0),
        attention_mask = encoded_text.attention_mask.flatten().unsqueeze(0),
        token_type_ids = encoded_text.token_type_ids.flatten().unsqueeze(0)
        )[-1]
    
    predicted_sentiment = torch.argmax(logit_output, 1)
    
    return convert_idx_to_sentiment(predicted_sentiment)



    
if __name__ == '__main__':
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    sentiment_model = FinBERT.load_from_checkpoint(r'checkpoints\best-checkpoint.ckpt')
    sentiment_model.freeze()
    
    get_sentiment('Apple Pay Later Could Pose Larger Threat To Card Issuers Than To BNPL Players')
