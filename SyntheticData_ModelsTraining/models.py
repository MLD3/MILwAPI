import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nystrom_attention import NystromAttention
import numpy as np
from clam_models import *

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


class ABDMIL(nn.Module):
    def __init__(self, currfeatureEx, M, E, L, D, PE = False):
        super(ABDMIL, self).__init__()

        self.M = M
        self.E = E
        self.L = L
        self.D = D
        self.K = 1   
        self.PE = PE
                
        self.ftEx = currfeatureEx(self.M, self.E, self.L)
        
            
        self.H3 = torch.zeros((10, self.L))
        d = self.H3.shape[1]
        for i in range(self.H3.shape[0]):
            for j in range(self.H3.shape[1]//2):
                weight = (1/1000)**((2*j)/d)
                self.H3[i, 2*j] = torch.sin(torch.tensor(weight*i))
                self.H3[i, 2*j + 1] = torch.cos(torch.tensor(weight*i))
                
        if PE == True:
            self.L = self.L * 2

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear((self.L)*self.K, 2)
        )

        self.intermediate_classifier = nn.Sequential(
            nn.Linear(self.L, 2)
        )

    def forward(self, x):
        # print(H3.dtype, H2.dtype)
        H = self.ftEx(x)
        
        if self.PE == True:
            H = torch.cat((H, self.H3.to(H.device)), 1)
        H = self.feature_extractor_part3(H)

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return A, Y_prob

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        y, attn = self.attn(self.norm(x), return_attn = True)
        x = x + y

        attn = attn[:, :,  -x.shape[1]:,  -x.shape[1]:]
        return x, attn


class TransMIL_end(nn.Module):
    def __init__(self, n_classes, dim, outerdim):
        super(TransMIL_end, self).__init__()
        self.pos_layer = PPEG(dim = dim)
        self._fc1 = nn.Sequential(nn.Linear(outerdim, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim = dim)
        self.layer2 = TransLayer(dim = dim)
        self.norm = nn.LayerNorm(dim)
        self._fc2 = nn.Linear(dim, self.n_classes)


    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]

        h = self._fc1(h) #[B, n, 512]
        orig_h = h.shape[1]

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h, first_attns = self.layer1(h) #[B, N, 512]

        # #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h, second_attns = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        return logits


class TransMIL(nn.Module):
    def __init__(self, currfeatureEx, M, E, L, D, PE = False):
        super(TransMIL, self).__init__()

        self.M = M
        self.E = E
        self.L = L
        self.D = D
        self.K = 1   
        self.PE = PE
                
        self.ftEx = currfeatureEx(self.M, self.E, self.L)
        
            
        self.H3 = torch.zeros((10, self.L))
        d = self.H3.shape[1]
        for i in range(self.H3.shape[0]):
            for j in range(self.H3.shape[1]//2):
                weight = (1/1000)**((2*j)/d)
                self.H3[i, 2*j] = torch.sin(torch.tensor(weight*i))
                self.H3[i, 2*j + 1] = torch.cos(torch.tensor(weight*i))
                
        if PE == True:
            self.L = self.L * 2

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )

        self.transmil_classifier = TransMIL_end(n_classes=2, dim = self.D, outerdim = self.L)

    def forward(self, x):
        # print(H3.dtype, H2.dtype)
        H = self.ftEx(x)
        
        if self.PE == True:
            H = torch.cat((H, self.H3.to(H.device)), 1)
        H = self.feature_extractor_part3(H)

        Y_prob = self.transmil_classifier(data = H.unsqueeze(0))

        return Y_prob
    
    
class DTFD_Tier1(nn.Module):
    def __init__(self, cL, cD, PE = False):
        super(DTFD_Tier1, self).__init__()
        self.PE = PE
        self.L = cL
        
        if PE == True:
            self.L = self.L * 2
            
        self.D = cD
        self.K = 1
    

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L, 2)
        )

    def forward(self, H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return M, Y_prob

class DTFD_Tier2(nn.Module):
    def __init__(self, cL, cD, PE = False):
        super(DTFD_Tier2, self).__init__()        
        
        self.L = cL
        self.D = cD
        self.K = 1
        
        if PE == True:
            self.L = self.L * 2
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2)
        )
        
    def forward(self, H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob
    
class DTFD(nn.Module):
    def __init__(self, currFeatureEx, cM, cE, cL, cD, num_pseudo_bags, PE = False):
        super(DTFD, self).__init__()
        self.M = cM
        self.E = cE
        self.PE = PE
        self.L = cL
        
        self.ftEx = currFeatureEx(self.M, self.E, self.L)
        
        self.num_pseudo_bags = num_pseudo_bags
        self.curr_DTFD_Tier1 = DTFD_Tier1(cL, cD, PE)
        self.curr_DTFD_Tier2 = DTFD_Tier2(cL, cD, PE)
        
        self.H3 = torch.zeros((10, self.L))
        d = self.H3.shape[1]
        for i in range(self.H3.shape[0]):
            for j in range(self.H3.shape[1]//2):
                weight = (1/1000)**((2*j)/d)
                self.H3[i, 2*j] = torch.sin(torch.tensor(weight*i))
                self.H3[i, 2*j + 1] = torch.cos(torch.tensor(weight*i))

    
    def forward(self, x, y_i, criterion):
        H = self.ftEx(x)
        if self.PE == True:
            H = torch.cat((H, self.H3.to(H.device)), 1)

        H_idxs = np.random.permutation(H.shape[0])
        loss_bag = 0
        fts = []
        preds1 = []
        inter_count = 0
        loss_bag = 0
        
        for i in range(0, H.shape[0], H.shape[0]//self.num_pseudo_bags):
            currH = H[H_idxs[i:i+H.shape[0]//self.num_pseudo_bags]]
            tier_1_ft, tier_1_prediction = self.curr_DTFD_Tier1(currH)
            fts.append(tier_1_ft.squeeze(0))
            if loss_bag == 0:
                loss_bag = criterion(tier_1_prediction, y_i.long())
            else:
                loss_bag += criterion(tier_1_prediction, y_i.long())
            inter_count += 1
        loss_bag = loss_bag/inter_count

        fts = torch.stack(fts)
        tier_2_prediction = self.curr_DTFD_Tier2(fts)
        loss_bag += criterion(tier_2_prediction, y_i.long())
        
        return tier_2_prediction, loss_bag    
    

class Transformer_End(nn.Module):
    def __init__(self, n_classes, dim, outerdim):
        super(Transformer_End, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(outerdim, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim = dim)
        self.layer2 = TransLayer(dim = dim)
        self.norm = nn.LayerNorm(dim)
        self._fc2 = nn.Linear(dim, self.n_classes)


    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]

        h = self._fc1(h) #[B, n, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h, first_attns = self.layer1(h) #[B, N, 512]

        #---->Translayer x2
        h, second_attns = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        return logits
    
    
class Transformer_GAP_End(nn.Module):
    def __init__(self, n_classes, dim, outerdim):
        super(Transformer_GAP_End, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(outerdim, dim), nn.ReLU())
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim = dim)
        self.layer2 = TransLayer(dim = dim)
        self.norm = nn.LayerNorm(dim)
        self._fc2 = nn.Linear(dim, self.n_classes)


    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]

        h = self._fc1(h) #[B, n, 512]

        #---->Translayer x1
        h, first_attns = self.layer1(h) #[B, N, 512]

        #---->Translayer x2
        h, second_attns = self.layer2(h) #[B, N, 512]

        #---->cls_token
        print(h.shape)
        h = torch.mean(self.norm(h), 1)

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        return logits

    
class Transformer(nn.Module):
    def __init__(self, currFeatureEx, cM, cE, cL, cD, PE = False):
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

        self.transformer_classifier = Transformer_End(n_classes=2, dim = self.D, outerdim = self.L)


    def forward(self, x):
        H = self.ftEx(x)
        if self.PE == True:
            H = torch.cat((H, self.H3.to(H.device)), 1)

        Y_prob = self.transformer_classifier(data = H.unsqueeze(0))

        return Y_prob
    
    
class Transformer_GAP(nn.Module):
    def __init__(self, currFeatureEx, cM, cE, cL, cD, PE = False):
        super(Transformer_GAP, self).__init__()

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

        self.transformer_classifier = Transformer_GAP_End(n_classes=2, dim = self.D, outerdim = self.L)


    def forward(self, x):
        H = self.ftEx(x)
        if self.PE == True:
            H = torch.cat((H, self.H3.to(H.device)), 1)

        Y_prob = self.transformer_classifier(data = H.unsqueeze(0))

        return Y_prob

class ClamWrapper(nn.Module):
    def __init__(self, currFeatureEx, cM, cE, cL, cK, cDrop, cSB, PE = False):
        super(ClamWrapper, self).__init__()
        
        self.PE = PE
        
        self.M = cM
        self.E = cE
        self.L = cL
        
        self.ftEx = currFeatureEx(self.M, self.E, self.L)
        
        self.currModel = CLAM_SB(k_sample = cK, dropout = cDrop, PE = PE)
        if cSB == False:
            self.currModel = CLAM_MB(k_sample = cK, dropout = cDrop, PE = PE)
            
        self.L = cL

        self.H3 = torch.zeros((10, self.L))
        d = self.H3.shape[1]
        for i in range(self.H3.shape[0]):
            for j in range(self.H3.shape[1]//2):
                weight = (1/1000)**((2*j)/d)
                self.H3[i, 2*j] = torch.sin(torch.tensor(weight*i))
                self.H3[i, 2*j + 1] = torch.cos(torch.tensor(weight*i))
                
    def forward(self, x, y, instance_eval):
        H = self.ftEx(x)
        
        if self.PE == True:
            H = torch.cat((H, self.H3.to(H.device)), 1)
        
        Y_prob, inst_dict = self.currModel(H, y, instance_eval)
        return Y_prob, inst_dict
        
                    
        

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