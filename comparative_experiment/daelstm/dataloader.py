import numpy as np
from numpy import linalg as la
import _pickle as cPickle
import torch
from torch.utils.data import DataLoader, TensorDataset



def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1

def to_amp_phase(X_train, X_test, nsamples):
    X_train_cmplx = X_train[:, 0, :] + 1j * X_train[:, 1, :]
    X_test_cmplx = X_test[:, 0, :] + 1j * X_test[:, 1, :]

    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:, 1, :], X_train[:, 0, :]) / np.pi

    X_train_amp = np.reshape(X_train_amp, (-1, 1, nsamples))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, nsamples))

    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    X_train = np.transpose(np.array(X_train), (0, 2, 1))


    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, 1, :], X_test[:, 0, :]) / np.pi

    X_test_amp = np.reshape(X_test_amp, (-1, 1, nsamples))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, nsamples))

    X_test = np.concatenate((X_test_amp, X_test_ang), axis=1)
    X_test = np.transpose(np.array(X_test), (0, 2, 1))
    return (X_train, X_test)


def norm_pad_zeros(X_train, nsamples):
    print("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i, :, 0] = X_train[i, :, 0] / la.norm(X_train[i, :, 0], 2)
    return X_train

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

    X_train, X_test = to_amp_phase(X_train, X_test, 128)
    for i in range(X_train.shape[0]):
        k = 2 / (X_train[i, :, 1].max() - X_train[i, :, 1].min())
        X_train[i, :, 1] = -1 + k * (X_train[i, :, 1] - X_train[i, :, 1].min())
    for i in range(X_test.shape[0]):
        k = 2 / (X_test[i, :, 1].max() - X_test[i, :, 1].min())
        X_test[i, :, 1] = -1 + k * (X_test[i, :, 1] - X_test[i, :, 1].min())
        

    X_SNR_train = list(map(lambda x: snrs.index(lbl[x][1]), train_idx))
    X_SNR_test = list(map(lambda x: snrs.index(lbl[x][1]), test_idx))
    Y_train = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = list(map(lambda x: mods.index(lbl[x][0]), test_idx))

    train_set_all = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train),torch.tensor(X_SNR_train))
    test_set_all = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test), torch.tensor(X_SNR_test))
    train_loader_all = DataLoader(dataset=train_set_all, batch_size=batch_size,shuffle=True)
    test_loader_all = DataLoader(dataset=test_set_all, batch_size=batch_size)
    return train_loader_all,test_loader_all