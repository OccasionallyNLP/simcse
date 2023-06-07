import torch
from tqdm import tqdm
import numpy as np
from utils.distributed_utils import *
import torch.nn.functional as F
from scipy.stats import spearmanr

def evaluation(model, dataloader):
    model.eval()
    Sentence1=[]
    Sentence2=[]
    Actual=[]
    with torch.no_grad():
        for data in dataloader:
            labels = data['labels']
            data = {i:j.cuda() for i,j in data.items() if i!='labels'}
            s1, s2 = model.forward(**data)
            Sentence1.append(s1)
            Sentence2.append(s2)
            Actual.extend(labels.cpu().tolist())
    Predict = F.cosine_similarity(torch.cat(Sentence1,0), torch.cat(Sentence2,0))
    cor, pvalue = spearmanr(Actual, Predict.cpu().tolist())
    return dict(cor=cor), Predict

def get_scores(local_rank, scores, distributed:bool):
    if distributed:
        total_corr = [j.item() for j in get_global(local_rank, torch.tensor([scores['cor']]).cuda())]
        total_corr = sum(total_corr)/len(total_corr)
    else:
        total_corr = scores['cor']
    return dict(corr=np.round(total_corr,3))
