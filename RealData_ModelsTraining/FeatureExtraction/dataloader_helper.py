import os
from os import listdir
from os.path import isfile, join
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

    
def dataloader_mimic(type_):
    X_path = []
    y_path = []
    with open('/data1/cxr_full/MIMIC/%s_labels.txt'%type_) as f:
        for row in f:
            X_path.append('/data1/cxr_full/MIMIC/%s/%s.jpg'%(type_, row.split(', ')[0]))
            y_path.append(float(row.split(', ')[1].split('\n')[0]))
            
    return X_path, y_path


def singleconvertAndLabel(mypath):
    image = Image.open(mypath)
    transform = transforms.PILToTensor()
    tensor = transform(image)
    transform = transforms.Resize(size=(512*6, 512*4))
    tensor = transform(tensor)

    if tensor.shape[0] > 1:
        comb_tensor = torch.cat((tensor[0].unsqueeze(0), tensor[0].unsqueeze(0), tensor[0].unsqueeze(0)), 0) 
    else:
        comb_tensor = torch.cat((tensor, tensor, tensor), 0)
    
    return comb_tensor