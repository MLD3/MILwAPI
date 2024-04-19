import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nystrom_attention import NystromAttention
import numpy as np

def lin_soft(ys):
    sum_exp = 0
    for y in ys:
        sum_exp += torch.exp(y)
    output = 0
    for y in ys:
        output += y*torch.exp(y)/sum_exp
    

    return output

def cust_bce(a, b,dim=None):
    if dim is None:
        return (b * (-torch.log(a+1e-8)) + (1-b) * (-torch.log(1-a+1e-8))).mean()
    elif dim==-1:
        return (b * (-torch.log(a+1e-8)) + (1-b) * (-torch.log(1-a+1e-8)))
    else:
        return (b * (-torch.log(a+1e-8)) + (1-b) * (-torch.log(1-a+1e-8))).flatten(dim).mean(dim)

def weight(x,y,a):
    x_min = x.min(1)[0]
    x_max = x.max(1)[0]
    x_mean = x.mean(1)
    x_median = x.median(1)[0]
    x = 2**torch.abs(x_max-x_median)/2

    y = y.view(-1,1,1,1) * torch.ones(a.shape).cuda()
    x = x.view(-1,1,1,1) * torch.ones(a.shape).cuda()
    x = torch.max(x,1-y)
    return x.detach()

def sgl(x,y, delta, pool_before_act, pooling = lin_soft):
    """
        Calculation of the self-guiding loss
        ________________________________________________________________

        x:                  Input predictions
        y:                  Bag-level Label
        pooling:            Pooling function to use
        delta:              skalar for upper/lower threshold
        pool_before_act:    boolean determining when to apply activation function
        ________________________________________________________________

        loss:               Image-level skalar loss
        inst:               Instance-level skalar loss
    """
    x = x.reshape(1, -1)
    rho = (x.sigmoid() - x.sigmoid().min(1,keepdim=True)[0])/(x.sigmoid().max(1,keepdim=True)[0] - x.sigmoid().min(1,keepdim=True)[0])
    rho = rho.detach()
    delta_l = delta
    delta_h = 1-delta

    mask = torch.ones(rho.shape).cuda() * -1
    mask[rho<delta_l] = 0
    mask[rho>delta_h] = 1
    mask = mask * y.view(-1,1,1,1)
    mask = mask.detach()
    

    if pool_before_act:
        loss = cust_bce(pooling(x.sigmoid()),y)
    else:
        loss = cust_bce(pooling(x).sigmoid(),y)

    
    weight_map = weight(x.sigmoid().detach(),y,mask)
    x = x.unsqueeze(0).unsqueeze(0)
    rho = rho.unsqueeze(0).unsqueeze(0)

    
    if (mask==-1).sum()==0 and (mask==1).sum()==0:
        inst = (cust_bce(x[mask==0].sigmoid(),0).sum()*weight_map[mask==0]).mean() 
    elif (mask==-1).sum()==0 and (mask==0).sum()==0:
        inst = (cust_bce(x[mask==1].sigmoid(),torch.ones(x[mask==1].shape).cuda())*weight_map[mask==1]).mean()
    elif (mask==-1).sum()==0:
        inst = (cust_bce(x[mask==1].sigmoid(),torch.ones(x[mask==1].shape).cuda())*weight_map[mask==1]).mean() + (cust_bce(x[mask==0].sigmoid(),0).sum()*weight_map[mask==0]).mean() 
    else:
        inst = (cust_bce(x[mask==1].sigmoid(),torch.ones(x[mask==1].shape).cuda())*weight_map[mask==1]).mean() +\
            (cust_bce(x[mask==0].sigmoid(),0).sum()*weight_map[mask==0]).mean()  +\
            (cust_bce(x[mask==-1].sigmoid(),rho[mask==-1])*weight_map[mask==-1]).mean()



    return loss, inst


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

