import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nystrom_attention import NystromAttention
import numpy as np
from das import *

class ftEx(nn.Module):
    def __init__(self, M, E, L):
        super(ftEx, self).__init__()

        self.M = M
        self.E = E

        self.L = L

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, self.M, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.M, self.E, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.E * 4 * 4, self.L),
            nn.ReLU(),
        )
        

    def forward(self, x):
        x = x.squeeze(0)

        H2 = self.feature_extractor_part1(x)
        H2 = H2.view(-1, self.E * 4 * 4)
        H2 = self.feature_extractor_part2(H2)  # NxL

        return H2
    
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
    def __init__(self, currFeatureEx, cM, cE, cL, cD):
        super(Transformer, self).__init__()
        
        self.M = cM
        self.E = cE
        self.L = cL
        self.D = cD
        
        self.ftEx = currFeatureEx(self.M, self.E, self.L)
    
            
        self.D = cD
        self.K = 1

        self.transformer_classifier = Transformer_End(n_classes=2, dim = self.D, outerdim = self.L)


    def forward(self, x):
        H = self.ftEx(x)

        Y_prob = self.transformer_classifier(data = H)

        return Y_prob
    
    

        

class CIFAR10Dataset(Dataset):
    def __init__(self, name, data):
        """Init GenomeDataset."""
        super().__init__()
        self.name = name
        self.features = data['features']
        self.numinstances =  data['features'].shape[1]
        self.labels = data['labels']
        self.labels2 = data['labels2']
        self.length = len(self.features)

    def __len__(self):
        """Return number of sequences."""
        return self.length

    def __getitem__(self, index):
        """Return sequence and label at index."""
        return index, self.features[index].float(), self.labels[index].float(), self.labels2[index*self.numinstances:index*self.numinstances+self.features.shape[1]]