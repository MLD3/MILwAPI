import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nystrom_attention import NystromAttention
import numpy as np

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
    def __init__(self, attn, norm_layer=nn.LayerNorm, dim=512, add = True):
        super(TransLayer, self).__init__()
        self.norm = norm_layer(dim)
        self.attntype = attn
        self.add = add
        if attn == 'Nystrom':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = dim//8,
                heads = 8,
                num_landmarks = dim//2,    # number of landmarks
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=0.1,
            )
        elif attn == 'Orig':
            self.attn = nn.MultiheadAttention(dim, 8, batch_first = True)

    def forward(self, x):
        if self.attntype == 'Nystrom':
            y, attn = self.attn(self.norm(x), return_attn = True)
        else:
            y, attn = self.attn(self.norm(x), self.norm(x), self.norm(x))
        if self.add == True:
            x = x + y
        else: 
            x = y
            
        return x, attn

class Transformer_End(nn.Module):
    def __init__(self, n_classes, dim, outerdim, agg, attn, add = True):
        super(Transformer_End, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(outerdim, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(attn, dim = dim, add = add)
        self.layer2 = TransLayer(attn, dim = dim, add = add)
        self.norm = nn.LayerNorm(dim)
        self._fc2 = nn.Linear(dim, self.n_classes)
        self.agg = agg

    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]

        h = self._fc1(h) #[B, n, 512]

        #---->cls_token
        if self.agg == 'cls_token':
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
            h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h, first_attns = self.layer1(h) #[B, N, 512]

        #---->Translayer x2
        h, second_attns = self.layer2(h) #[B, N, 512]

        #---->cls_token
        if self.agg == 'cls_token':
            h = self.norm(h)[:,0]
        elif self.agg == 'avg':
            h = torch.mean(self.norm(h), 1)
        elif self.agg == 'max':
            h, _ = torch.max(self.norm(h), 1)
            
        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        return logits
    
    
class Transformer(nn.Module):
    def __init__(self, currFeatureEx, cM, cE, cL, cD, agg, attn, add = True, PE = False):
        super(Transformer, self).__init__()

        self.PE = PE
        
        self.M = cM
        self.E = cE
        self.L = cL
        self.D = cD
        
        self.ftEx = currFeatureEx(self.M, self.E, self.L)
        
        self.H3 = torch.zeros((10, self.L))
        d = self.H3.shape[1]
        for i in range(self.H3.shape[0]):
            for j in range(self.H3.shape[1]//2):
                weight = (1/1000)**((2*j)/d)
                self.H3[i, 2*j] = torch.sin(torch.tensor(weight*i))
                self.H3[i, 2*j + 1] = torch.cos(torch.tensor(weight*i))

        
        if PE == True:
            self.L = self.L * 2
            
        self.D = cD
        self.K = 1

        self.transformer_classifier = Transformer_End(n_classes=2, dim = self.D, outerdim = self.L, agg = agg, attn = attn, add = add)


    def forward(self, x):
        H = self.ftEx(x)
        if self.PE == True:
            H = torch.cat((H, self.H3.to(H.device)), 1)

        Y_prob = self.transformer_classifier(data = H.unsqueeze(0), PE = self.H3.to(H.device))

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