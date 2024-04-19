import torch
import os
from dataloader_helper import *
from models import *
import numpy as np
import random
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import joblib

torch.backends.cudnn.deterministic = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train, y_train = np.array(joblib.load('/data2/meerak/MIMIC_CXR_FTS/UPDATED_train_fts.joblib')), np.array(joblib.load('/data2/meerak/MIMIC_CXR_FTS/UPDATED_train_ys.joblib'))
X_val, y_val = np.array(joblib.load('/data2/meerak/MIMIC_CXR_FTS/UPDATED_val_fts.joblib')), np.array(joblib.load('/data2/meerak/MIMIC_CXR_FTS/UPDATED_val_ys.joblib'))
X_test, y_test = np.array(joblib.load('/data2/meerak/MIMIC_CXR_FTS/UPDATED_test_fts.joblib')), np.array(joblib.load('/data2/meerak/MIMIC_CXR_FTS/UPDATED_test_ys.joblib'))

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    
dg = CXRDataset('train', {'features':torch.tensor(X_train), 'labels':torch.tensor(y_train)})
val_dg = CXRDataset('val', {'features':torch.tensor(X_val), 'labels':torch.tensor(y_val)})
test_dg = CXRDataset('test', {'features':torch.tensor(X_test), 'labels':torch.tensor(y_test)})

train_loader = DataLoader(dg,batch_size = 1,shuffle = True)
val_loader = DataLoader(val_dg,batch_size = 1,shuffle = False)
test_loader = DataLoader(test_dg,batch_size = 1,shuffle = False)

# D:1024, LR:1e-5, WD:1e-6, NPB:24
cD = 1024
cLR = 1e-4
cWD = 1e-6
cNPB = 4
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

model = DTFD(cD, cNPB, PE = True).to(device)

optimizer = optim.Adam(model.parameters(), lr=cLR, weight_decay = cWD)
#LOSS FUNCTION
criterion = nn.CrossEntropyLoss()

train_losses = []
train_aurocs = []
val_losses = []
val_aurocs = []
test_losses = []
test_aurocs = []

max_acc = 0
stop_idx = 0

for epoch in range(100):
    losses = []
    bag_preds = []
    bag_ys = []

    print('Epoch:', epoch)

    for batch_idx, (curridx, data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.to(device).squeeze(0)
        target = target.to(device)

        bag_prediction, loss_bag = model(data, target, criterion)

        bag_preds.extend(F.softmax(bag_prediction, dim = 1)[:, 1].detach().cpu().numpy())
        bag_ys.extend(target.detach().cpu().numpy())

        loss_bag.backward()
        optimizer.step()

        losses.append(loss_bag.detach().cpu().item())
        if batch_idx % 100 == 0:
            print('train', batch_idx, losses[-1])
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
    for batch_idx, (curridx, data, target) in enumerate(val_loader):
        optimizer.zero_grad()

        data = data.to(device).squeeze(0)
        target = target.to(device)

        bag_prediction, loss_bag = model(data, target, criterion)

        bag_preds.extend(F.softmax(bag_prediction, dim = 1)[:, 1].detach().cpu().numpy())
        bag_ys.extend(target.detach().cpu().numpy())
        losses.append(loss_bag.detach().cpu().item())
        if batch_idx % 100 == 0:
            print('val', batch_idx, losses[-1])
        del data
        del bag_prediction
        del loss_bag

    bag_preds = np.array(bag_preds)
    bag_ys = np.array(bag_ys)  
    print('Val AUROC:', roc_auc_score(bag_ys, bag_preds))
    val_losses.append(sum(losses)/len(losses))
    val_aurocs.append(roc_auc_score(bag_ys, bag_preds))

    losses = []
    bag_preds = []
    bag_ys = []
    for batch_idx, (curridx, data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        data = data.to(device).squeeze(0)
        target = target.to(device)

        bag_prediction, loss_bag = model(data, target, criterion)

        bag_preds.extend(F.softmax(bag_prediction, dim = 1)[:, 1].detach().cpu().numpy())
        bag_ys.extend(target.detach().cpu().numpy())
        losses.append(loss_bag.detach().cpu().item())
        del data
        del bag_prediction
        del loss_bag

    bag_preds = np.array(bag_preds)
    bag_ys = np.array(bag_ys)  
    print('Test AUROC:', roc_auc_score(bag_ys, bag_preds))
    test_losses.append(sum(losses)/len(losses))
    test_aurocs.append(roc_auc_score(bag_ys, bag_preds))

    torch.save(model.state_dict(), '/data2/meerak/models/best_dtfd_pe_mimiccxr_densenet_epoch%d'%(epoch))

    if val_aurocs[-1] < max_acc:
        stop_idx += 1
    elif val_aurocs[-1] >= max_acc:
        max_acc = val_aurocs[-1]
        stop_idx = 0
    if stop_idx >= 5 and epoch > 10:
        break

    with open('best_dtfd_pe.txt', 'w') as f:
        count = 0
        f.write('tl, ta, vl, va, ta\n')
        f.write('D:%d, LR:1e%d, WD:1e%d, NPB:%d\n'%(cD, np.log10(cLR), np.log10(cWD), cNPB))
        for tl, ta, vl, va, testa in zip(train_losses, train_aurocs, val_losses, val_aurocs, test_aurocs):
            f.write('%d, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n'%(count, tl, ta, vl, va, testa))
            count += 1





