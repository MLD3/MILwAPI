import torch
import os
from dataloader_helper import *
from ft_ex import *
import numpy as np
import random
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import joblib


torch.backends.cudnn.deterministic = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_labels, y_train = dataloader_mimic('train')
X_val_labels, y_val = dataloader_mimic('val')
X_test_labels, y_test = dataloader_mimic('test')

model = featureEX_sup().to(device)
model.load_state_dict(torch.load('../models/UPDATED_supervised_patch_pretraining_chexpert_epoch2_hyp0'))

train_fts = []
train_ys = []

for batch_idx in range(len(X_train_labels)):
    data = singleconvertAndLabel(X_train_labels[batch_idx]).unsqueeze(0).float().to(device)
    target = torch.tensor(y_train[batch_idx])
    bag_prediction, _ = model(data)
    if batch_idx % 10 == 0:
        print(batch_idx, bag_prediction.detach().cpu().numpy().shape)
    train_fts.append(bag_prediction.detach().cpu().numpy())
    train_ys.append(target.detach().cpu().numpy())
    
joblib.dump(train_fts, 'UPDATED_train_fts.joblib')
joblib.dump(train_ys, 'UPDATED_train_ys.joblib')

val_fts = []
val_ys = []

for batch_idx in range(len(X_val_labels)):
    data = singleconvertAndLabel(X_val_labels[batch_idx]).unsqueeze(0).float().to(device)
    target = torch.tensor(y_val[batch_idx])
    bag_prediction, _ = model(data)

    if batch_idx % 10 == 0:
        print(batch_idx, bag_prediction.detach().cpu().numpy().shape)
    val_fts.append(bag_prediction.detach().cpu().numpy())
    val_ys.append(target.detach().cpu().numpy())

joblib.dump(val_fts, 'UPDATED_val_fts.joblib')
joblib.dump(val_ys, 'UPDATED_val_ys.joblib')

test_fts = []
test_ys = []

for batch_idx in range(len(X_test_labels)):
    data = singleconvertAndLabel(X_test_labels[batch_idx]).unsqueeze(0).float().to(device)
    bag_prediction, _ = model(data)
    target = torch.tensor(y_test[batch_idx])
    
    if batch_idx % 10 == 0:
        print(batch_idx, bag_prediction.detach().cpu().numpy().shape)
    test_fts.append(bag_prediction.detach().cpu().numpy())
    test_ys.append(target.detach().cpu().numpy())

joblib.dump(test_fts, 'UPDATED_test_fts.joblib')
joblib.dump(test_ys, 'UPDATED_test_ys.joblib')




