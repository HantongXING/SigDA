from dataloader import dataloader_10a_m2sfe
from model import Vgg2 as Vgg
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')


current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_root = os.path.join(current_dir, '../checkpoints', 'vgg2.pth')
dataset_root = os.path.join(
    current_dir, '../../Datasets', 'RML2016.10a_dict.pkl')

train_loader_all, test_loader_all = dataloader_10a_m2sfe(path=dataset_root,
                                                         batch_size=400, train_ratio=0.8)


model = Vgg()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


precesion_previous = 0

for epoch in range(1600):
    print('Epoch: {}, learn_rate: {:.8f}'.format(
        epoch, optimizer.param_groups[0]['lr']))
    for i, data in enumerate(train_loader_all):
        train_data, train_label, traon_snr = data
        train_data, train_label = train_data.cuda(), train_label.cuda()
        logits, _ = model(train_data)
        loss = criterion(logits, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 40 == 0:
            print('Epoch-{}  iter-{}  Loss:{:.6f}'.format(
                epoch, i,
                loss.detach().cpu().cuda(),
            ))

    model.eval()
    correct = 0
    fals = 0
    cmt = torch.zeros(11, 11, 20, dtype=torch.int16)
    for i, data in enumerate(test_loader_all):
        testdata, testlabel, SNR = data
        testdata, testlabel = testdata.cuda(), testlabel.cuda()
        outputs, _ = model(testdata)

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
    print(
        'Best Accuracy: {:.4f}, from Epoch {} / {}'.format(precesion_previous, best_epoch, epoch))
