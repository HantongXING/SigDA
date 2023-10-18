import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')


from model.m2sfe import M2SFE
from utils.dataloader import dataloader_10a_m2sfe
from utils.utils import mask

############################## Create dataloader ###########################################
path = '/data/xht/modulation_recog/RML2016.10a_dict.pkl'
train_loader_all,test_loader_all = dataloader_10a_m2sfe(path,batch_size=64,train_ratio=0.8)


############################# Testing #############################################
net = torch.load('/data/xht/modulation_recog/open-source-code/checkpoints/source.pth')
net.eval()
correct,fals = 0,0
cmt = torch.zeros(11, 11, 20, dtype=torch.int16)

for i, data in enumerate(test_loader_all):
    testdata, testlabel, SNR = data        
    testdata, testlabel = testdata.cuda(), testlabel.cuda()
    outputs,_,_ = net(testdata)
    _, predict = torch.max(outputs, 1)
    for k in range(len(predict)):
        if predict[k] == testlabel[k]:
            correct = correct + 1
        else:
            fals = fals + 1
        cmt[testlabel[k]][predict[k]][SNR[k]] = cmt[testlabel[k]][predict[k]][SNR[k]] + 1

print(correct / (correct + fals))
for j in range(20):
    num = 0
    print(cmt[:, :, j])
    for k in range(11):
        num = num + cmt[k, k, j]
    print(int(num) / int(sum(sum(cmt[:, :, j]))))




