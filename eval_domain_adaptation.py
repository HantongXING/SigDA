from utils.da_loader import data_loader
import torch
from model.config import config
import warnings
import os
warnings.filterwarnings('ignore')


############################## Create dataloader ###########################################
current_dir = os.path.dirname(os.path.abspath(__file__))
data_root_10A = os.path.join(
    current_dir, config.data_root, config.data_name_10A)
data_root_04C = os.path.join(
    current_dir, config.data_root, config.data_name_04C)
checkpoint_path_backbone = os.path.join(
    current_dir, config.model_root, config.checkpoints_DA)

train_loader_all, test_loader_all = data_loader(
    path_04c=data_root_04C,
    path_10a=data_root_10A,
    batch_size=config.Batch_size)


net = torch.load(checkpoint_path_backbone)
net.eval()
correct, fals = 0, 0
cmt = torch.zeros(11, 11, 20, dtype=torch.int16)
for i, data in enumerate(test_loader_all):
    testdata, testlabel, SNR = data
    testdata = testdata.cuda()
    testlabel = testlabel.cuda()
    outputs, _, _ = net(testdata)
    _, predict = torch.max(outputs, 1)
    for k in range(len(predict)):
        if predict[k] == testlabel[k]:
            correct = correct + 1
        else:
            fals = fals + 1
        cmt[testlabel[k]][predict[k]][SNR[k]
                                      ] = cmt[testlabel[k]][predict[k]][SNR[k]] + 1

print('Testing accuracy: {:.4f}'.format(correct / (correct + fals)))
for j in range(20):
    num = 0
    if config.show_confusion_metrix:
        print(cmt[:, :, j])
    for k in range(11):
        num = num + cmt[k, k, j]
    print(int(num) / int(sum(sum(cmt[:, :, j]))))
