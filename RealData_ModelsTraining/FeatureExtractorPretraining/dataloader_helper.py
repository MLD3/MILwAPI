import os
from os import listdir
from os.path import isfile, join
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import csv

class CXRDataset(Dataset):
    def __init__(self, name, data):
        """Init GenomeDataset."""
        super().__init__()
        self.name = name
        self.features = data['features']
        self.numinstances =  data['features'].shape[1]
        self.labels = data['labels']
        self.length = len(self.features)

    def __len__(self):
        """Return number of sequences."""
        return self.length

    def __getitem__(self, index):
        """Return sequence and label at index."""
        return index, self.features[index].float(), self.labels[index].float()
    

def get_data(path):
    X_pos, y_pos = convertAndLabel(path + '/Cardiomegaly')
    X_neg, y_neg = convertAndLabel(path + '/No_Cardiomegaly')
    
    X = torch.cat((X_pos, X_neg), 0)
    y = torch.cat((y_pos, y_neg))
    
    return X, y


def dataloader_chexpert():
    start = '/data1/cxr_full/Chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/'
    with open(start + 'train.csv', newline='') as csvfile:
        count = 0
        spamreader = csv.reader(csvfile)
        dict_labels = {}
        dict_header = []
        necessary_index = 0
        for row in spamreader:
            if count == 0:
                dict_header = row
                necessary_index = dict_header.index('Cardiomegaly')
            if count > 0:
                if row[3] == 'Frontal':
                    if row[necessary_index] == '1.0' or row[necessary_index] == '0.0':
                        key = row[0].split('/')[2] + row[0].split('/')[3] + row[0].split('/')[4]
                        if row[necessary_index] == '1.0':
                            dict_labels[key] = 1
                        else:
                            dict_labels[key] = 0
            count += 1

    train_path = '/data1/cxr_full/Chexpert/chexpertchestxrays-u20210408/my_splits/pretrain_train'
    val_path = '/data1/cxr_full/Chexpert/chexpertchestxrays-u20210408/my_splits/pretrain_val'

    trainfiles = [f for f in listdir(train_path) if isfile(join(train_path, f))]
    valfiles = [f for f in listdir(val_path) if isfile(join(val_path, f))]

    X_train_path = []
    y_train_path = []

    X_val_path = []
    y_val_path = []
    for elem in trainfiles:
        X_train_path.append(train_path + '/' + elem)
        y_train_path.append(dict_labels[elem])

    for elem in valfiles:
        X_val_path.append(val_path + '/' + elem)
        y_val_path.append(dict_labels[elem])

    return X_train_path, y_train_path, X_val_path, y_val_path

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

def convertAndLabel(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # print(onlyfiles)

    tensors = []

    count = 0
    for file in onlyfiles:
        image = Image.open(mypath + '/' + file)
        transform = transforms.PILToTensor()
        tensor = transform(image)
        transform = transforms.Resize(size=(512*6, 512*4))
        tensor = transform(tensor)
        
        if tensor.shape[0] > 1:
            comb_tensor = torch.cat((tensor[0].unsqueeze(0), tensor[0].unsqueeze(0), tensor[0].unsqueeze(0)), 0) 
            tensors.append(comb_tensor)
        else:
            comb_tensor = torch.cat((tensor, tensor, tensor), 0)
            tensors.append(comb_tensor)
        count += 1
        if count % 50 == 0:
            print(count, tensor.shape)

    test = torch.stack(tensors)
    ys = []
    if '/Cardiomegaly' in mypath:
        ys = [1]*len(onlyfiles)
    elif '/No_Cardiomegaly' in mypath:
        ys = [0]*len(onlyfiles)
        
    return test, torch.tensor(ys)