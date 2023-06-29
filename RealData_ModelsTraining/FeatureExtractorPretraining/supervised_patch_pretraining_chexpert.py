import torch
import os
from dataloader_helper import *
from models_densenet import *
import numpy as np
import random
import torch.optim as optim
from sklearn.metrics import roc_auc_score


torch.backends.cudnn.deterministic = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_labels, y_train, X_val_labels, y_val = dataloader_chexpert()
cLR = 1e-5
cWD = 1e-6
IDX = 0
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

model = featureEX_sup().to(device)

optimizer = optim.Adam(model.parameters(), lr=cLR, weight_decay = cWD)
#LOSS FUNCTION
criterion = nn.CrossEntropyLoss()

train_losses = []
train_aurocs = []
val_losses = []
val_aurocs = []

max_acc = 0
stop_idx = 0

for epoch in range(3):
    losses = []
    bag_preds = []
    bag_ys = []

    print('Epoch:', epoch)

    count = -1
    for batch_idx in np.random.permutation(len(X_train_labels)):
        count += 1
        optimizer.zero_grad()

        data = singleconvertAndLabel(X_train_labels[batch_idx]).unsqueeze(0).float().to(device)
        target = torch.tensor(y_train[batch_idx]).repeat(24).to(device)

        bag_prediction = model(data)
        loss_bag = criterion(bag_prediction, target.long())

        bag_preds.extend(F.softmax(bag_prediction, dim = 1)[:, 1].detach().cpu().numpy())
        bag_ys.extend(target.detach().cpu().numpy())

        loss_bag.backward()
        optimizer.step()

        if count % 10 == 0:
            print('train', count, bag_ys[-10:], bag_preds[-10:])

        losses.append(loss_bag.cpu().item())
        del data
        del bag_prediction
        del loss_bag

    bag_preds = np.array(bag_preds)
    bag_ys = np.array(bag_ys)  
    print('Train AUROC:', roc_auc_score(bag_ys, bag_preds))
    train_losses.append(sum(losses)/len(losses))
    train_aurocs.append(roc_auc_score(bag_ys, bag_preds))

    losses = []
    bag_preds = []
    bag_ys = []
    count = -1
    for batch_idx in range(len(X_val_labels)):
        count += 1
        optimizer.zero_grad()
        data = singleconvertAndLabel(X_val_labels[batch_idx]).unsqueeze(0).float().to(device)
        target = torch.tensor(y_val[batch_idx]).repeat(24).to(device)

        bag_prediction = model(data)

        loss_bag = criterion(bag_prediction, target.long())

        if count % 10 == 0:
            print('val', count, bag_ys[-10:], bag_preds[-10:])

        bag_preds.extend(F.softmax(bag_prediction, dim = 1)[:, 1].detach().cpu().numpy())
        bag_ys.extend(target.detach().cpu().numpy())
        losses.append(loss_bag.cpu().item())
        del data
        del bag_prediction
        del loss_bag

    bag_preds = np.array(bag_preds)
    bag_ys = np.array(bag_ys)  
    print('Val AUROC:', roc_auc_score(bag_ys, bag_preds))
    val_losses.append(sum(losses)/len(losses))
    val_aurocs.append(roc_auc_score(bag_ys, bag_preds))


    torch.save(model.state_dict(), '../models/UPDATED_supervised_patch_pretraining_chexpert_epoch%d_hyp%d'%(epoch, IDX))

    # if val_aurocs[-1] < max_acc:
    #     stop_idx += 1
    # elif val_aurocs[-1] >= max_acc:
    #     max_acc = val_aurocs[-1]
    #     stop_idx = 0
    # if stop_idx == 30:
    #     break

    with open('UPDATED_hyp%d_supervised_patch_pretraining_chexpert.txt'%IDX, 'w') as f:
        count = 0
        f.write('tl, ta, vl, va, ta\n')
        for tl, ta, vl, va in zip(train_losses, train_aurocs, val_losses, val_aurocs):
            f.write('%d, %0.4f, %0.4f, %0.4f, %0.4f\n'%(count, tl, ta, vl, va))
            count += 1






