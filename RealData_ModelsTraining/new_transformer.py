import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from das import *

class TransLayer(nn.Module):
    def __init__(self, dim, outerdim, norm_layer=nn.LayerNorm):
        super(TransLayer, self).__init__()
        self.attn = MultiHeadDistanceAwareSelfAttention(outerdim, dim, dim)
        self.dim = dim

    def forward(self, x):
        edge_idx = torch.tensor(np.array([[i] for i in range(x.shape[0])])).to(x.device)
        edge_attr = torch.tensor(np.array([[i/100] for i in range(x.shape[0])])).to(x.device).float()
        y = self.attn(x, edge_idx, edge_attr)
            
        return y

class Transformer_End(nn.Module):
    def __init__(self, n_classes, dim, outerdim):
        super(Transformer_End, self).__init__()
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim = dim, outerdim = outerdim)
        self.norm = nn.LayerNorm(dim)
        self._fc2 = nn.Linear(dim, self.n_classes)

    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->cls_token
        h, _ = torch.max(self.norm(h), 0)
        
        #---->predict
        logits = self._fc2(h).unsqueeze(0) #[B, n_classes]
        return logits
    
    
class Transformer(nn.Module):
    def __init__(self, cD):
        super(Transformer, self).__init__()
        
        self.D = cD
        self.K = 1

        self.transformer_classifier = Transformer_End(n_classes=2, dim = self.D, outerdim = 1000)


    def forward(self, H):

        Y_prob = self.transformer_classifier(data = H)

        return Y_prob
    
    

        

