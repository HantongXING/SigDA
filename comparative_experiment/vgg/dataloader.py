import numpy as np
import _pickle as cPickle
import torch
from torch.utils.data import DataLoader, TensorDataset



def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


def dataloader_10a_m2sfe(path='/data/xht/modulation_recog/RML2016.10a_dict.pkl',
                         batch_size=64,
                         train_ratio=0.8):

    a = open(path, 'rb')
    Xd = cPickle.load(a, encoding='iso-8859-1')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    for mod in mods:
        for snr  in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
    X = np.vstack(X)
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = np.uint32(n_examples * train_ratio)

    train_idx = list(np.random.choice(range(0, n_examples), size=n_train, replace=False))
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]
    X_SNR_train = list(map(lambda x: snrs.index(lbl[x][1]), train_idx))
    X_SNR_test = list(map(lambda x: snrs.index(lbl[x][1]), test_idx))
    # Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    # Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    Y_train = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = list(map(lambda x: mods.index(lbl[x][0]), test_idx))

    train_set_all = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train),torch.tensor(X_SNR_train))
    test_set_all = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test), torch.tensor(X_SNR_test))
    train_loader_all = DataLoader(dataset=train_set_all, batch_size=batch_size,shuffle=True)
    test_loader_all = DataLoader(dataset=test_set_all, batch_size=batch_size)
    return train_loader_all,test_loader_all