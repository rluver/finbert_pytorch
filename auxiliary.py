# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:26:09 2021

@author: MJH
"""


def categorizer(label):
    
    if label == 'positive':
        return 2
    elif label == 'neutral':
        return 1
    elif label == 'negative':
        return 0



def convert_idx_to_sentiment(idx):
    
    return ['negative', 'neutral', 'positive'][idx]