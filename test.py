# test T5
# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import time
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import numpy as np
from transformers import AutoTokenizer, T5Tokenizer, T5EncoderModel, T5Config
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from utils.evaluation import *
from model import *
import pandas as pd
from matplotlib import pyplot as plt

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--test_data', type=str, help = 'test_data 위치')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    # PTM model
    parser.add_argument('--check_point_dir', type=str)
    
    # 데이터 관련
    parser.add_argument('--max_length',type= int, default = 512)
    parser.add_argument('--batch_size', default = 8, type=int)
    
    # TODO
    ## distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    with open(os.path.join(args.check_point_dir,'args.txt'), 'r') as f:
        check_point_args = json.load(f)    
    ###########################################################################################
    # tokenizer, config, model
    ###########################################################################################
    config = T5Config.from_pretrained(check_point_args['ptm_path'])
    tokenizer = AutoTokenizer.from_pretrained(check_point_args['ptm_path'])
    model_type = T5EncoderModel
    model = SentenceModel(config, model_type) 
    model.load_state_dict(torch.load(os.path.join(check_point_args['output_dir'],'best_model')))
    
    ###########################################################################################
    # device
    ###########################################################################################
    # TODO
    if args.local_rank == -1:  # single-node multi-gpu (or cpu) mode
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    # multi gpu
    else:
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(device) 
        model.to(device)
    ###########################################################################################
    
    ###########################################################################################
    # data
    ###########################################################################################
    test_data = load_jsonl(args.test_data)
    
    test_dataset = STSDataset(test_data, tokenizer, args.max_length)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, sampler = test_sampler, collate_fn = test_dataset.collate_fn)
    
    scores_,_ = evaluation(model, test_dataloader)
    scores = get_scores(args.local_rank, scores_, args.distributed)
    print(scores)
    ###########################################################################################
    