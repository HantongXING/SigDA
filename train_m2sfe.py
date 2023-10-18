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


################################ Create model ##############################################
model = M2SFE()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss(reduce=True, size_average = True)



################################## Training ###############################################
precesion_previous=0
for epoch in range(300):  
    for i, data in enumerate(train_loader_all):
        train_data, train_label,train_snr = data
        feed_data = mask(train_data)
        train_data,feed_data = train_data.cuda(), feed_data.cuda()
        train_label = train_label.cuda()

        logits,_,gen_data = model(feed_data) 

        loss_ce = criterion1(logits, train_label)
        loss_mse = criterion2(gen_data,train_data)
        loss = loss_ce + loss_mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if i % 100 == 0:
            print('Epoch-{}  iter-{}  Loss:{:.6f}    Loss-ce:{:.6f}   Loss-mse:{:.6f}'.format(
                epoch, i, 
                loss.detach().cpu().cuda(),
                loss_ce.detach().cpu().cuda(),
                loss_mse.detach().cpu().cuda()
                ))
    
            
    ############################## Valadating ########################################  
    model.eval()
    correct,fals = 0,0
    cmt = torch.zeros(11, 11, 20, dtype=torch.int16)

    for i, data in enumerate(test_loader_all):
        testdata, testlabel, SNR = data        
        testdata, testlabel = testdata.cuda(), testlabel.cuda()
        outputs,_,_ = model(testdata)
        _, predict = torch.max(outputs, 1)
        for k in range(len(predict)):
            if predict[k] == testlabel[k]:
                correct = correct + 1
            else:
                fals = fals + 1
            cmt[testlabel[k]][predict[k]][SNR[k]] = cmt[testlabel[k]][predict[k]][SNR[k]] + 1
    model.train()
    
    precision = correct / (correct + fals)
    print('Epoch: {}, Validation accuracy: {:.4f}'.format(epoch, precision))
    if precision>precesion_previous:
        print('accuracy improved...save model...')
        best_epoch = epoch
        torch.save(model, '/data/xht/modulation_recog/open-source-code/checkpoints/m2sfe.pth')
        precesion_previous = precision
        for j in range(20):
            num = 0
            for k in range(11):
                num = num + cmt[k, k, j]
            print(int(num) / int(sum(sum(cmt[:, :, j]))))
    print('Best accuracy now: {:.4f}, From epoch: {} / {}'.format(
        precesion_previous, best_epoch, epoch))


############################# Testing #############################################
net = torch.load('/data/xht/modulation_recog/open-source-code/checkpoints/m2sfe.pth')
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

print('Testing accuracy: {:.4f}'.format(correct / (correct + fals)))
for j in range(20):
    num = 0
    print(cmt[:, :, j])
    for k in range(11):
        num = num + cmt[k, k, j]
    print(int(num) / int(sum(sum(cmt[:, :, j]))))




