import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from nystrom_attention import NystromAttention
import numpy as np
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from token_transformer import Token_transformer
from token_performer import Token_performer
from transformer_block import Block, get_sinusoid_encoding
from clam_models import *
    
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
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, add = True):
        super(TransLayer, self).__init__()
        self.norm = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first = True)

    def forward(self, x):
        y, attn = self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = x + y
        return x, attn


class TransMIL_End(nn.Module):
    def __init__(self, n_classes, dim, outerdim):
        super(TransMIL_End, self).__init__()
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
    
    

class TransMIL_Orig(nn.Module):
    def __init__(self, cD):
        super(TransMIL_Orig, self).__init__()        
        self.L = 1000
        
        self.D = cD
        self.K = 1


        self.transmil_classifier = TransMIL_End(n_classes=2, dim = self.D, outerdim = self.L)


    def forward(self, H):
        Y_prob = self.transmil_classifier(data = H.unsqueeze(0))

        return Y_prob

    