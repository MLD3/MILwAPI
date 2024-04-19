from models import *
import os
import joblib
import random
import numpy as np
torch.backends.cudnn.enabled=False
from sklearn.metrics import roc_auc_score


torch.backends.cudnn.deterministic = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sparsity = 1/10
bag_size = 10
for num_bags in [100, 1000, 10000]:
    X_train = joblib.load('/home/user/MNIST/X_train_%d_%d_%d'%((int(sparsity*100), bag_size, num_bags)))
    y_train = joblib.load('/home/user/MNIST/y_train_%d_%d_%d'%((int(sparsity*100), bag_size, num_bags)))
    actual_y_train = joblib.load('/home/user/MNIST/actual_y_train_%d_%d_%d'%((int(sparsity*100), bag_size, num_bags)))

    X_val = joblib.load('/home/user/MNIST/X_val_%d_%d_%d'%((int(sparsity*100), bag_size, num_bags)))
    y_val = joblib.load('/home/user/MNIST/y_val_%d_%d_%d'%((int(sparsity*100), bag_size, num_bags)))
    actual_y_val = joblib.load('/home/user/MNIST/actual_y_val_%d_%d_%d'%((int(sparsity*100), bag_size, num_bags)))

    X_test = joblib.load('/home/user/MNIST/X_test_%d_%d_%d'%((int(sparsity*100), bag_size, 250)))
    y_test = joblib.load('/home/user/MNIST/y_test_%d_%d_%d'%((int(sparsity*100), bag_size, 250)))
    actual_y_test = joblib.load('/home/user/MNIST/actual_y_test_%d_%d_%d'%((int(sparsity*100), bag_size, 250)))

    dg = CIFAR10Dataset('train', {'features':torch.tensor(X_train), 'labels':torch.tensor(y_train), 'labels2':torch.tensor(actual_y_train)})
    val_dg = CIFAR10Dataset('val', {'features':torch.tensor(X_val), 'labels':torch.tensor(y_val), 'labels2':torch.tensor(actual_y_val)})
    test_dg = CIFAR10Dataset('test', {'features':torch.tensor(X_test), 'labels':torch.tensor(y_test), 'labels2':torch.tensor(actual_y_test)})

    train_loader = DataLoader(dg,batch_size = 1,shuffle = True)
    val_loader = DataLoader(val_dg,batch_size = 1,shuffle = False)
    test_loader = DataLoader(test_dg,batch_size = 1,shuffle = False)

    for perm_dist in range(0, 11):
        train_shuffle_idxs = joblib.load('/home/user/MNIST/train_idxs_numbags_%d_permdist_%d.joblib'%(num_bags, perm_dist))
        val_shuffle_idxs = joblib.load('/home/user/MNIST/val_idxs_numbags_%d_permdist_%d.joblib'%(num_bags, perm_dist))
        test_shuffle_idxs = joblib.load('/home/user/MNIST/test_idxs_permdist_%d.joblib'%perm_dist)

        cM = 40
        cE = 100
        cL = 500

        #  K:1, Drop:1, cC:0.5, LR:1e-4, WD:1e-7
        cLR = 1e-4
        cWD = 1e-7

        cK = 1
        cDrop = 1
        cC = 0.5

        for seed in range(10):
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            model = ClamWrapper(ftEx, cM, cE, cL, cK, cDrop, False, PE = False).to(device)
            optimizer = optim.Adam(model.parameters(), lr=cLR, weight_decay = cWD)
            #LOSS FUNCTION
            criterion = nn.CrossEntropyLoss()

            train_losses = []
            val_losses = []
            val_aurocs = []
            test_aurocs = []

            max_acc = 0
            stop_idx = 0

            for epoch in range(500):
                losses = []
                count = -1

                for curridx, x_i, y_i, actual_y_i in train_loader:
                    count += 1
                    optimizer.zero_grad()
                    x_i = x_i.transpose(0, 1)
                    y_i = y_i.to(device) 

                    x_i = x_i[train_shuffle_idxs[curridx]]
                    actual_y_i = actual_y_i.squeeze(0)[train_shuffle_idxs[curridx]]

                    # positive pair, with encoding
                    bag_prediction, inst_dict = model(x_i.to(device), y_i.to(torch.int64), instance_eval = True)
                    loss_bag = cC*criterion(bag_prediction, y_i.long())
                    loss_bag += (1-cC)*inst_dict['instance_loss']

                    loss_bag.backward()
                    optimizer.step()

                    losses.append(loss_bag.detach().cpu().item())
                    del x_i
                    del bag_prediction
                    del loss_bag


                train_losses.append(sum(losses)/len(losses))

                print('epoch', epoch, 'train loss', train_losses[-1])

                losses = []
                count = -1
                bag_preds = []
                bag_ys = []

                for curridx, x_i, y_i, actual_y_i in val_loader:
                    count += 1
                    x_i = x_i.transpose(0, 1)
                    optimizer.zero_grad()
                    y_i = y_i.to(device)

                    x_i = x_i[val_shuffle_idxs[curridx]]
                    actual_y_i = actual_y_i.squeeze(0)[val_shuffle_idxs[curridx]]

                    # positive pair, with encoding
                    bag_prediction, inst_dict = model(x_i.to(device), y_i.to(torch.int64), instance_eval = True)
                    loss_bag = cC*criterion(bag_prediction, y_i.long())
                    loss_bag += (1-cC)*inst_dict['instance_loss']

                    bag_preds.extend(F.softmax(bag_prediction, dim = 1)[:, 1].detach().cpu().numpy())
                    bag_ys.extend(y_i.detach().cpu().numpy())
                    losses.append(loss_bag.detach().cpu().item())
                    del x_i
                    del bag_prediction
                    del loss_bag

                bag_preds = np.array(bag_preds)
                bag_ys = np.array(bag_ys)  

                val_losses.append(sum(losses)/len(losses))
                val_aurocs.append(roc_auc_score(bag_ys, bag_preds))

                print('epoch', epoch, 'val loss', val_losses[-1])
                print('epoch', epoch, 'val auroc', val_aurocs[-1])

                count = -1
                bag_preds = []
                bag_ys = []

                for curridx, x_i, y_i, actual_y_i in test_loader:
                    count += 1
                    x_i = x_i.transpose(0, 1)
                    optimizer.zero_grad()
                    y_i = y_i.to(device)

                    x_i = x_i[test_shuffle_idxs[curridx]]
                    actual_y_i = actual_y_i.squeeze(0)[test_shuffle_idxs[curridx]]

                    # positive pair, with encoding
                    bag_prediction, inst_dict = model(x_i.to(device), y_i.to(torch.int64), instance_eval = True)
                    loss_bag = cC*criterion(bag_prediction, y_i.long())
                    loss_bag += (1-cC)*inst_dict['instance_loss']

                    bag_preds.extend(F.softmax(bag_prediction, dim = 1)[:, 1].detach().cpu().numpy())
                    bag_ys.extend(y_i.detach().cpu().numpy())

                    del x_i
                    del bag_prediction

                bag_preds = np.array(bag_preds)
                bag_ys = np.array(bag_ys)

                curr_rocs = []
                for _ in range(1000):
                    idxs = np.random.choice(len(bag_preds), len(bag_preds))
                    curr_bag_preds = bag_preds[idxs]
                    curr_bag_ys = bag_ys[idxs]
                    curr_rocs.append(roc_auc_score(curr_bag_ys, curr_bag_preds))
                curr_rocs.sort()

                test_aurocs.append(curr_rocs[500])

                print('epoch', epoch, 'bag test auroc', test_aurocs[-1])

                if val_aurocs[-1] < max_acc:
                    stop_idx += 1
                elif val_aurocs[-1] >= max_acc:
                    max_acc = val_aurocs[-1]
                    stop_idx = 0
                if stop_idx >= 5 and epoch > 10:
                    break

                with open('/data2/user/txtfiles/seed%d_numbags%d_MNIST_permdist%d_ClamMB_loss.txt'%(seed, num_bags, perm_dist), 'w') as f:
                    count = -1
                    f.write('epoch, train loss, val loss, val auroc, test aurocs\n')
                    for tl, vl, va, testa in zip(train_losses, val_losses, val_aurocs, test_aurocs):
                        count += 1
                        f.write('%d, %0.2f, %0.2f, %0.10f, %0.10f\n'%(count, tl, vl, va, testa))
