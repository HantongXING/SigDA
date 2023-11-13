import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings
warnings.filterwarnings('ignore')


import torch.optim as optim
import torch.nn as nn
import torch

from model.domainclassifer import DomainClassifer
from utils.da_loader import data_loader

############################## Create dataloader ###########################################
train_loader_all,test_loader_all = data_loader(
        path_04c='/data/xht/modulation_recog/2016.04C.multisnr.pkl',
        path_10a='/data/xht/modulation_recog//RML2016.10a_dict.pkl',
        batch_size=64)


################################ Create model ##############################################
domainclassifer = DomainClassifer()
domainclassifer = domainclassifer.cuda()
domainclassifer=torch.nn.DataParallel(domainclassifer)

MS = torch.load('/data/xht/modulation_recog/open-source-code/checkpoints/m2sfe.pth')
MS.eval()
MT = torch.load('/data/xht/modulation_recog/open-source-code/checkpoints/m2sfe.pth')
print(MT)

optimizerd = optim.Adam(domainclassifer.parameters(), lr=0.0001)
optimizerg = optim.Adam(MT.parameters(), lr=0.00001)

criterion = nn.CrossEntropyLoss()

################################## Training ###############################################
precesion_previous=0
for epoch in range(300):
    for i, (sdata, tdata) in enumerate(zip(train_loader_all, test_loader_all)):
        
        train_data, _,_ = sdata
        test_data, _, _ = tdata
        
        train_data,test_data = train_data.cuda(),test_data.cuda()

        l_one = torch.ones(test_data.shape[0])
        l_zero = torch.zeros(train_data.shape[0])
        
        _,train_feature,_ = MS(train_data)
        _,test_feature,_ = MT(test_data)        

        D_loss = criterion(domainclassifer(train_feature.detach()), l_zero.long().cuda()) + criterion(domainclassifer(test_feature.detach()), l_one.long().cuda())
        optimizerd.zero_grad()
        D_loss.backward()
        optimizerd.step()

        loss1 = criterion(domainclassifer(test_feature), l_zero.long().cuda())
        G_loss = loss1
        optimizerg.zero_grad()
        G_loss.backward()
        optimizerg.step()
        
        
        if i % 100 == 0:
            print('Epoch-{}  iter-{}  Loss-D:{:.6f}    Loss-G:{:.6f}'.format(
                epoch, i, 
                D_loss.detach().cpu().cuda(),
                G_loss.detach().cpu().cuda(),        
                ))

            
    ############################## Valadating ######################################## 
    MT.eval()
    correct,fals = 0,0
    cmt = torch.zeros(11, 11, 20, dtype=torch.int16)

    for i, data in enumerate(test_loader_all):
        testdata, testlabel, SNR = data
        testdata = testdata.cuda()
        testlabel = testlabel.cuda()
        outputs,_,_ = MT(testdata)
        _, predict = torch.max(outputs, 1)
        for k in range(len(predict)):
            if predict[k] == testlabel[k]:
                correct = correct + 1
            else:
                fals = fals + 1
            cmt[testlabel[k]][predict[k]][SNR[k]] = cmt[testlabel[k]][predict[k]][SNR[k]] + 1
    MT.train()

    precision = correct / (correct + fals)
    print('Epoch: {}, Validation accuracy: {:.4f}'.format(epoch, precision))
    if precision>precesion_previous:
        best_epoch = epoch
        torch.save(MT, '/data/xht/modulation_recog/open-source-code/checkpoints/target_model.pth')
        torch.save(domainclassifer, '/data/xht/modulation_recog/open-source-code/checkpoints/domainclassifer.pth')
        precesion_previous = precision
    for j in range(10):
        num = 0
        for k in range(11):
            num = num + cmt[k, k, j+10]
        print(int(num) / int(sum(sum(cmt[:, :, j+10]))))
    print('Best accuracy now: {:.4f}, From epoch: {} / {}'.format(
        precesion_previous, best_epoch, epoch))



############################# Testing #############################################
net = torch.load('/data/xht/modulation_recog/open-source-code/checkpoints/target_model.pth')
net.eval()
correct,fals = 0,0
cmt = torch.zeros(11, 11, 20, dtype=torch.int16)
for i, data in enumerate(test_loader_all):
    testdata, testlabel, SNR = data
    testdata = testdata.cuda()
    testlabel = testlabel.cuda()
    outputs,_,_ = net(testdata)
    _, predict = torch.max(outputs, 1)
    for k in range(len(predict)):
        if predict[k] == testlabel[k]:
            correct = correct + 1
        else:
            fals = fals + 1
        cmt[testlabel[k]][predict[k]][SNR[k]] = cmt[testlabel[k]][predict[k]][SNR[k]] + 1

print('Testing accuracy: {:.4f}'.format(correct / (correct + fals)))
for j in range(20):
    num = 0
    print(cmt[:,:,j])
    for k in range(11):
        num = num + cmt[k, k, j]
    print(int(num) / int(sum(sum(cmt[:, :, j]))))


