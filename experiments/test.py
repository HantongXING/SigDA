import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('/data/xht/modulation_recog/open-source-code/model')


from torch.utils.data import DataLoader, TensorDataset
import torch
import _pickle as cPickle
import numpy as np
from autoencoder import DNET

import warnings
warnings.filterwarnings('ignore')

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


############################## Create dataset #########################################
a = open('/data/xht/modulation_recog/RML2016.10a_dict.pkl', 'rb')
Xd = cPickle.load(a, encoding='iso-8859-1')
snrs, mods = map(lambda j: sorted(
    list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)
np.random.seed(2016)
n_examples = X.shape[0]
n_train = np.uint32(n_examples * 0.8)

train_idx = list(np.random.choice(
    range(0, n_examples), size=n_train, replace=False))
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]
X_SNR_train = list(map(lambda x: snrs.index(lbl[x][1]), train_idx))
X_SNR_test = list(map(lambda x: snrs.index(lbl[x][1]), test_idx))
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

train_set_all = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
test_set_all = TensorDataset(torch.tensor(
    X_test), torch.tensor(Y_test), torch.tensor(X_SNR_test))
train_loader_all = DataLoader(
    dataset=train_set_all, batch_size=64, shuffle=True)
test_loader_all = DataLoader(dataset=test_set_all, batch_size=128)


############################## Testing #############################################
net = torch.load(
    '/data/xht/modulation_recog/open-source-code/checkpoints/source.pth')
net.eval()
correct = 0
fals = 0
cmt = torch.zeros(11, 11, 20, dtype=torch.int16)
for i, data in enumerate(test_loader_all):
    testdata, testlabel, SNR = data
    testdata, testlabel = testdata.cuda(), testlabel.cuda()
    outputs, _ = net(testdata)
    _, label = torch.max(testlabel, 1)
    _, predict = torch.max(outputs, 1)
    for k in range(len(predict)):
        if predict[k] == label[k]:
            correct = correct + 1
        else:
            fals = fals + 1
        cmt[label[k]][predict[k]][SNR[k]] = cmt[label[k]][predict[k]][SNR[k]] + 1

print(correct / (correct + fals))
for j in range(20):
    num = 0
    print(cmt[:, :, j])
    for k in range(11):
        num = num + cmt[k, k, j]
    print(int(num) / int(sum(sum(cmt[:, :, j]))))
