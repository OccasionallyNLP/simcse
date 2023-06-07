import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import tensor as T
from transformers import PreTrainedModel

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class SentenceModel(PreTrainedModel):
    def __init__(self, config, model_class, pool='mean', temp=0.05):
        super().__init__(config)
        self.pretrained_model = model_class(config)
        self.pool = pool
        self.cos_sim = Similarity(temp)
        
    def init_pretrained_model(self, state_dict):
        # init pretrained model
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # input_ids : (bs, num_sent, seq_len)
        bs, num_sent, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len) # bs*num_sent, seq_len
        attention_mask = attention_mask.reshape(-1, seq_len)
        # output : (bs*num_sent, seq_len, dim)
        output = self.pretrained_model(input_ids, attention_mask) 
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                out = output.last_hidden_state[:,0,:] # bs*num_sent, dim
            else:
                out = output['pooler_output'] # bs*num_sent, dim
        elif self.pool == 'mean':
            out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, dim
            out = out/(s)
        # out : (bs*num_sent, dim)
        out = out.reshape(bs, num_sent, -1) # bs, num_sent, dim
        if num_sent == 2:
            z1, z2 = out[:,0],out[:,1] # sentence1, sentence2
            return z1,z2
        elif num_sent == 3:
            z1, z2, z3 = out[:,0],out[:,1],out[:,2] # anchor, positive, negative
            if 'labels' in kwargs:
                labels = kwargs['labels']
                c1 = self.cos_sim(z1.unsqueeze(1),z2.unsqueeze(0)) # bs, bs
                c2 = self.cos_sim(z1.unsqueeze(1),z3.unsqueeze(0)) # bs, bs
                c =  torch.cat([c1,c2],dim=1) # bs, 2*bs
                loss = F.cross_entropy(c, labels)
                return dict(loss = loss)
                
            else:
                return z1, z2, z3
