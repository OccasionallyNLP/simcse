# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import logging
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel, T5Config
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from model import *
from utils.evaluation import *

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, help = 'test_name')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    # data
    parser.add_argument('--train_data', type=str, help = 'train_data 위치')
    parser.add_argument('--val_data', type=str, help='val data 위치')
    # logging 관련
    parser.add_argument('--logging_term', type=int, default = 100)
   
    # 학습 관련
    parser.add_argument('--epochs', default = 10, type=int)
    parser.add_argument('--eval_epoch', type = int, default = 1, help = 'term of evaluation')
    parser.add_argument('--batch_size', default = 8, type=int)
    parser.add_argument('--lr', type=float, default = 5e-5)
    parser.add_argument('--warmup', type=float, default = 1000)
    parser.add_argument('--decay', type=float, default = 0.05)
    parser.add_argument('--fp16', type=str2bool, default = False)
    parser.add_argument('--accumulation_steps', type=int, default = 1) # 221124 추가
    
    # PTM model
    parser.add_argument('--ptm_path', type=str)
    parser.add_argument('--model_path', type=str)
    
    # model input
    parser.add_argument('--max_length', type=int)
    
    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    parser.add_argument('--early_stop', type=str2bool, default = True) # XXX220919
    parser.add_argument('--patience', type=int, default = 3)
    args  = parser.parse_args()
    return args


def train():
    # optimizer
    optimizer_grouped_parameters = make_optimizer_group(model, args.decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.decay)
    # scheduler
    scheduler = get_linear_scheduler(len(train_dataloader)*args.epochs, args.warmup, optimizer, train_dataloader)
    if args.local_rank in [-1,0]:
        early_stop = EarlyStopping(args.patience, args.output_dir, max = True, min_difference=1e-5)
    if args.fp16:
        scaler = GradScaler()
    flag_tensor = torch.zeros(1).cuda()
    ########################################################################################
    # train
    ########################################################################################
    global_step = 0
    train_plot = []
    val_plot = []
    for epoch in range(1, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        Loss = 0.
        step = 0
        iter_bar = tqdm(train_dataloader, desc='step', disable=args.local_rank not in [-1,0])
        #train
        for data in iter_bar:
            optimizer.zero_grad()            
            data = {i:j.cuda() for i,j in data.items()}
            labels = torch.arange(data['input_ids'].size(0),device = 'cuda' if torch.cuda.is_available else 'cpu') 
            data['labels'] = labels
            if args.fp16:
                with autocast():
                    loss = model.forward(**data)['loss']
                    loss = loss / args.accumulation_steps
                    scaler.scale(loss).backward()
                    if (step+1)%args.accumulation_steps==0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
            else:
                loss = model.forward(**data)['loss']
                loss = loss / args.accumulation_steps
                loss.backward()
                if (step+1)%args.accumulation_steps==0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()
            step+=1
            scheduler.step()
            if args.distributed:
                torch.distributed.reduce(loss, 0)
                loss = loss / torch.distributed.get_world_size()
            Loss+=loss.item()
            iter_bar.set_postfix({'epoch':epoch, 'global_step':global_step, 'lr':f"{scheduler.get_last_lr()[0]:.5f}",'total_loss':f'{Loss/step:.5f}'}) # 감소한다는 것을 확인하는 것임.
            if global_step%args.logging_term == 0:
                if args.local_rank in [-1,0]:
                    logger1.info(iter_bar)
                    logger2.info(iter_bar)
            global_step+=1
            
        # epoch 당 기록.
        if args.local_rank in [-1,0]:
            logger1.info(iter_bar)
            logger2.info(iter_bar)
        ########################################################################################
        # evaluation
        ###################################################################################################
        if args.eval_epoch!=0 and epoch%args.eval_epoch==0:
            # validation
            val_scores_, _ = evaluation(model, val_dataloader)
            val_scores = get_scores(args.local_rank, val_scores_, args.distributed)            
            if args.local_rank in [-1,0]:
                logger1.info(f'Val ---- epoch : {epoch} ----- scores:{val_scores}')
                logger2.info(f'Val ---- epoch : {epoch} ----- scores:{val_scores}')
                model_to_save = model.module if hasattr(model,'module') else model
                early_stop.check(model_to_save, val_scores['corr'])  
                if early_stop.timetobreak:
                    flag_tensor += 1
            if args.distributed:
                torch.distributed.broadcast(flag_tensor, 0) 
                torch.distributed.barrier()
        ###################################################################################################
        if args.early_stop:    
            if flag_tensor:
                if args.local_rank in [-1,0]:
                    logger1.info('early stop')
                    logger2.info('early stop')
                break
    # 저장시 - gpu 0번 것만 저장 - barrier 필수
    if args.local_rank in [-1,0]:
        torch.save(early_stop.best_model, os.path.join(early_stop.save_dir,'best_model'))
        logger1.info('train_end')
        logger2.info('train end')

def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ptm_path)
    config = T5Config.from_pretrained(args.ptm_path)
    model = SentenceModel(config, T5EncoderModel)
    if args.model_path is None:
        t5 = T5EncoderModel.from_pretrained(args.ptm_path)
        model.init_pretrained_model(t5.state_dict())
    else:
        model_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
    return tokenizer, model 

def load_datasets(args, tokenizer):
    # LOAD DATASETS
    train_data = load_jsonl(args.train_data)
    train_dataset = NLIDataset(train_data, tokenizer, args.max_length)
    if args.distributed:
        # OK - legacy
        val_data = load_data(args.val_data, args.local_rank, args.distributed)
    else:
        val_data = load_jsonl(args.val_data)
    val_dataset = STSDataset(val_data, tokenizer, args.max_length)
    return train_dataset, val_dataset
        
        
if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok = True)
    if args.local_rank in [-1,0]:
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    logger1, logger2 = get_log(args)
    if args.local_rank in [-1,0]:
        logger1.info(args)
        logger2.info(args)
        
    ########################################################################################
    # tokenizer, model load
    ########################################################################################
    tokenizer, model = get_tokenizer_and_model(args)
    ########################################################################################
    
    ########################################################################################
    # distributed 관련
    ########################################################################################
    if args.distributed:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count()>1
        # 이 프로세스가 어느 gpu에 할당되는지 명시
        torch.cuda.set_device(args.local_rank)
        # 통신을 위한 초기화
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device = args.local_rank)
    else:
        model.cuda()
    ########################################################################################
    
    ########################################################################################
    # data
    ########################################################################################
    train_dataset, val_dataset = load_datasets(args, tokenizer)
    if args.distributed:
            train_sampler = DistributedSampler(train_dataset) 
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size, sampler = train_sampler, collate_fn = train_dataset.collate_fn)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size = args.batch_size, sampler = val_sampler, collate_fn = val_dataset.collate_fn)
    ########################################################################################
    
    ########################################################################################
    # train
    ########################################################################################
    train()
    ########################################################################################
