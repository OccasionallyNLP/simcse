# -*- coding: utf-8 -*-
# data_utils
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import random
import copy
from dataclasses import dataclass
from transformers import AutoTokenizer
from itertools import combinations

@dataclass
class NLIDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        inputs = []
        for i in batch:
            inputs.append(i['anchor'])
            inputs.append(i['positive'])
            inputs.append(i['negative'])
            
        inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        
        input_ids = inputs.input_ids.reshape(len(batch), 3, -1)
        attention_mask = inputs.attention_mask.reshape(len(batch), 3, -1)
        return dict(input_ids = input_ids, attention_mask = attention_mask)        
        
@dataclass
class STSDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        inputs = []
        labels = []
        for i in batch:
            inputs.append(i['sentence1'])
            inputs.append(i['sentence2'])
            labels.append(i['score'])
 
        inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        
        input_ids = inputs.input_ids.reshape(len(batch), 2, -1)
        attention_mask = inputs.attention_mask.reshape(len(batch), 2, -1)
        return dict(input_ids = input_ids, attention_mask = attention_mask, labels=torch.tensor(labels))        
        