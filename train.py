# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:21:35 2021

@author: MJH
"""


from finbert import FinBERT
from auxiliary import categorizer

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup



class FinancialPhraseBankDataset(Dataset):
    
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            text_max_token_length: int = 512
            ):
                
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_length = text_max_token_length
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index: int):
        
        data_row = self.data.iloc[index]
    
        encoded_text = self.tokenizer.encode_plus(
            data_row.phrase,
            max_length = self.text_max_token_length, 
            padding = 'max_length',
            truncation = True, 
            return_attention_mask = True, 
            add_special_tokens = True, 
            return_tensors = 'pt'
            )
        
        return dict(
            input_ids = encoded_text.input_ids.flatten(),
            attention_mask = encoded_text.attention_mask.flatten(),
            token_type_ids = encoded_text.token_type_ids.flatten(),
            label = torch.tensor(data_row.sentiment).unsqueeze(0)
            )
    
    
    
class FinancialPhraseBankDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: BertTokenizer,
        batch_size: int = 64,
        text_max_token_length: int = 512,
    ):
        
        super().__init__()
        
        self.train_df = train_df
        self.test_df = test_df
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_length = text_max_token_length
        
        self.setup()
        
        
    def __len__(self):
        return len(self.train_df)
        


    def setup(self, stage = None):
        self.train_dataset = FinancialPhraseBankDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_length,
            )
        
        self.test_dataset = FinancialPhraseBankDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_length,
            )
    
    
    def train_dataloader(self):        
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )

    
    def val_dataloader(self):        
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )
    
    
    def test_dataloader(self):        
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )
    
    

def main():
    
    financial_phrase_dataset = pd.read_csv('dataset/financial_phrase_bank/all-data.csv', encoding = 'latin-1', names = ['sentiment', 'phrase']).drop_duplicates().dropna().reset_index(drop = True)
    financial_phrase_dataset.sentiment = financial_phrase_dataset.sentiment.apply(lambda x: categorizer(x))
    train, test = train_test_split(financial_phrase_dataset, test_size = 0.3, shuffle = True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    EPOCHS = 10
    BATCH_SIZE = 32
    NUM_LABELS = 3
    LEARNING_RATE = 2e-5    
    DISCRIMINATIVE_FINE_TUNING_RATE = 0.85
        
    data_module = FinancialPhraseBankDataModule(train, test, tokenizer, batch_size = BATCH_SIZE)    
    model = FinBERT(model_path = 'model', train_samples = len(data_module), batch_size = BATCH_SIZE, epochs = EPOCHS, num_labels = NUM_LABELS, learning_rate = LEARNING_RATE, discriminative_fine_tuning_rate = DISCRIMINATIVE_FINE_TUNING_RATE)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = 'checkpoints',
        filename = 'best-checkpoint',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
        )

    logger = TensorBoardLogger('lightning_logs', name = 'finbert_sentiment')
  
    trainer = pl.Trainer(
        logger = logger,
        checkpoint_callback = checkpoint_callback,
        max_epochs = EPOCHS,
        gpus = 1,
        progress_bar_refresh_rate = 1
        )
    
    trainer.fit(model, data_module)
    
    
    
    
if __name__ == '__main__':
    main()
