from dataloader import dataloader_10a_m2sfe
from model import Dae6 as Dae
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')


current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_root = os.path.join(current_dir, '../checkpoints', 'daelstm.pth')
dataset_root = os.path.join(
    current_dir, '../../Datasets', 'RML2016.10a_dict.pkl')


train_loader_all, test_loader_all = dataloader_10a_m2sfe(path=dataset_root,
                                                         batch_size=128, train_ratio=0.8)

model = Dae()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss(reduce=True, size_average=True)


precesion_previous = 0
best_epoch = 0
for epoch in range(1600):
    print('Epoch: {}, learn_rate: {:.8f}'.format(
        epoch, optimizer.param_groups[0]['lr']))
    for i, data in enumerate(train_loader_all):
        train_data, train_label, train_snr = data
        train_data, train_label = train_data.cuda(), train_label.cuda()
        logits, decoder, _ = model(train_data)
        loss_ce = criterion1(logits, train_label)
        loss_mse = criterion2(decoder, train_data)
        loss = loss_ce*0.1 + loss_mse*0.9
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 80 == 0:
            print('Epoch-{}\titer-{}/{}\tLoss:{:.6f}\tLoss-ce:{:.6f}\tLoss-mse:{:.6f}'.format(
                epoch, i, len(train_loader_all),
                loss.detach().cpu().cuda(),
                loss_ce.detach().cpu().cuda(),
                loss_mse.detach().cpu().cuda()
            ))

    model.eval()
    correct = 0
    fals = 0
    cmt = torch.zeros(11, 11, 20, dtype=torch.int16)
    for i, data in enumerate(test_loader_all):
        testdata, testlabel, SNR = data
        testdata, testlabel = testdata.cuda(), testlabel.cuda()
        outputs, decoder, _ = model(testdata)
        _, predict = torch.max(outputs, 1)
        for k in range(len(predict)):
            if predict[k] == testlabel[k]:
                correct = correct + 1
            else:
                fals = fals + 1
            cmt[testlabel[k]][predict[k]][SNR[k]
                                          ] = cmt[testlabel[k]][predict[k]][SNR[k]] + 1
    model.train()

    precision = correct / (correct + fals)
    print('Epoch: {}, Validation accuracy: {:.4f}'.format(epoch, precision))
    if precision > precesion_previous:
        best_epoch = epoch
        precesion_previous = precision
        print('acc improved...save model...')
        torch.save(model, checkpoint_root)
        for j in range(20):
            num = 0
            for k in range(11):
                num = num + cmt[k, k, j]
            print(int(num) / int(sum(sum(cmt[:, :, j]))))
    print('Best accuracy now: {:.4f}, From epoch: {} / {}'.format(
        precesion_previous, best_epoch, epoch))
