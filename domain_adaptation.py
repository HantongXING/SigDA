from model.config import config
from utils.da_loader import data_loader
from model.domainclassifer import DomainClassifer
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
warnings.filterwarnings('ignore')

def run_da():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root_10A = os.path.join(
        current_dir, config.data_root, config.data_name_10A)
    data_root_04C = os.path.join(
        current_dir, config.data_root, config.data_name_04C)
    checkpoint_path = os.path.join(
        current_dir, config.model_root, config.checkpoints_10A)
    checkpoint_path_backbone = os.path.join(
        current_dir, config.model_root, config.checkpoints_DA)
    checkpoint_path_disc = os.path.join(
        current_dir, config.model_root, config.checkpoints_DA_disc)

    train_loader_all, test_loader_all = data_loader(
        path_04c=data_root_04C,
        path_10a=data_root_10A,
        batch_size=config.Batch_size)


    domainclassifer = DomainClassifer()
    domainclassifer = domainclassifer.cuda()
    domainclassifer = torch.nn.DataParallel(domainclassifer)

    MS = torch.load(checkpoint_path)
    MS.eval()
    MT = torch.load(checkpoint_path)
    print(MT)

    optimizerd = optim.Adam(domainclassifer.parameters(), lr=config.da_lr_d)
    optimizerg = optim.Adam(MT.parameters(), lr=config.da_lr_g)
    criterion = nn.CrossEntropyLoss()


    precesion_previous = 0
    best_epoch = 0
    for epoch in range(config.Epoch_num):
        for i, (sdata, tdata) in enumerate(zip(train_loader_all, test_loader_all)):

            train_data, _, _ = sdata
            test_data, _, _ = tdata

            train_data, test_data = train_data.cuda(), test_data.cuda()

            l_one = torch.ones(test_data.shape[0])
            l_zero = torch.zeros(train_data.shape[0])

            _, train_feature, _ = MS(train_data)
            _, test_feature, _ = MT(test_data)

            D_loss = criterion(domainclassifer(train_feature.detach()), l_zero.long().cuda(
            )) + criterion(domainclassifer(test_feature.detach()), l_one.long().cuda())
            optimizerd.zero_grad()
            D_loss.backward()
            optimizerd.step()

            loss1 = criterion(domainclassifer(test_feature), l_zero.long().cuda())
            G_loss = loss1
            optimizerg.zero_grad()
            G_loss.backward()
            optimizerg.step()

            if i % 200 == 0:
                print('Epoch-{}  iter-{}/{}  Loss-D:{:.6f}    Loss-G:{:.6f}'.format(
                    epoch, i, len(test_loader_all),
                    D_loss.detach().cpu().cuda(),
                    G_loss.detach().cpu().cuda(),
                ))

        ############################## Valadating ########################################
        MT.eval()
        correct, fals = 0, 0
        cmt = torch.zeros(11, 11, 20, dtype=torch.int16)

        for i, data in enumerate(test_loader_all):
            testdata, testlabel, SNR = data
            testdata = testdata.cuda()
            testlabel = testlabel.cuda()
            outputs, _, _ = MT(testdata)
            _, predict = torch.max(outputs, 1)
            for k in range(len(predict)):
                if predict[k] == testlabel[k]:
                    correct = correct + 1
                else:
                    fals = fals + 1
                cmt[testlabel[k]][predict[k]][SNR[k]
                                            ] = cmt[testlabel[k]][predict[k]][SNR[k]] + 1
        MT.train()

        precision = correct / (correct + fals)
        print('Epoch: {}, Validation accuracy: {:.4f}'.format(epoch, precision))
        if precision > precesion_previous:
            best_epoch = epoch
            torch.save(MT, checkpoint_path_backbone)
            torch.save(domainclassifer, checkpoint_path_disc)
            precesion_previous = precision
        for j in range(20):
            if config.show_confusion_metrix:
                print(cmt[:, :, j])
            num = 0
            for k in range(11):
                num = num + cmt[k, k, j]
            print(int(num) / int(sum(sum(cmt[:, :, j]))))
        print('Best accuracy now: {:.4f}, From epoch: {} / {}'.format(
            precesion_previous, best_epoch, epoch))

if __name__ == "__main__":
    run_da()