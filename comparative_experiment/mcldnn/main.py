from dataloader import dataloader_10a_m2sfe
from model import Mcldnn
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')


current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_root = os.path.join(current_dir, '../checkpoints', 'mcldnn.pth')
dataset_root = os.path.join(
    current_dir, '../../Datasets', 'RML2016.10a_dict.pkl')

############################## Create dataloader ###########################################
train_loader_all, test_loader_all = dataloader_10a_m2sfe(path=dataset_root,
                                                         batch_size=400, train_ratio=0.8)

################################ Create model ##############################################
model = Mcldnn()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

################################## Training ###############################################
precesion_previous = 0
best_epoch = 0
for epoch in range(1600):
    print('Epoch: {}, learn_rate: {:.8f}'.format(
        epoch, optimizer.param_groups[0]['lr']))
    for i, data in enumerate(train_loader_all):
        train_data, train_label, train_snr = data
        train_real = train_data[:, 0, :].unsqueeze(1)
        train_imag = train_data[:, 1, :].unsqueeze(1)
        train_data = train_data.unsqueeze(1)
        train_data, train_label = train_data.cuda(), train_label.cuda()
        train_real, train_imag = train_real.cuda(), train_imag.cuda()

        logits, _ = model(train_data, train_real, train_imag)
        loss = criterion(logits, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 40 == 0:
            print('Epoch-{}  iter-{}  Loss:{:.6f}'.format(
                epoch, i,
                loss.detach().cpu().cuda(),
            ))

    ############################## Valadating ########################################
    model.eval()
    correct, fals = 0, 0
    cmt = torch.zeros(11, 11, 20, dtype=torch.int16)
    for i, data in enumerate(test_loader_all):
        testdata, testlabel, SNR = data
        test_real = testdata[:, 0, :].unsqueeze(1)
        test_imag = testdata[:, 1, :].unsqueeze(1)
        testdata = testdata.unsqueeze(1)

        testdata, testlabel = testdata.cuda(), testlabel.cuda()
        test_real, test_imag = test_real.cuda(), test_imag.cuda()

        outputs, _ = model(testdata, test_real, test_imag)
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
