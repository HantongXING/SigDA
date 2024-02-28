import json
from model.config import config
from utils.dataloader import dataloader_10a_m2sfe
import warnings
import torch
import os
warnings.filterwarnings('ignore')


def run_eval():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, config.data_root, config.data_name_10A)
    _, test_loader_all = dataloader_10a_m2sfe(
        data_root, batch_size=config.Batch_size, train_ratio=0.8)

    checkpoint_path_10A = os.path.join(
        current_dir, config.model_root, config.checkpoints_10A)
  
    net = torch.load(checkpoint_path_10A)
    net.eval()
    correct, fals = 0, 0
    cmt = torch.zeros(11, 11, 20, dtype=torch.int16)

    for i, data in enumerate(test_loader_all):
        testdata, testlabel, SNR = data
        testdata, testlabel = testdata.cuda(), testlabel.cuda()
        outputs, _, _ = net(testdata)
        _, predict = torch.max(outputs, 1)
        for k in range(len(predict)):
            if predict[k] == testlabel[k]:
                correct = correct + 1
            else:
                fals = fals + 1
            cmt[testlabel[k]][predict[k]][SNR[k]
                                          ] = cmt[testlabel[k]][predict[k]][SNR[k]] + 1

    print(correct / (correct + fals))
    for j in range(20):
        num = 0
        if config.show_confusion_metrix:
            print(cmt[:, :, j])
        for k in range(11):
            num = num + cmt[k, k, j]
        print(int(num) / int(sum(sum(cmt[:, :, j]))))


if __name__ == '__main__':
    run_eval()
