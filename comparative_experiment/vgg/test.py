from dataloader import dataloader_10a_m2sfe
import torch
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings('ignore')


current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_root = os.path.join(current_dir, '../checkpoints', 'vgg2.pth')
dataset_root = os.path.join(
    current_dir, '../../Datasets', 'RML2016.10a_dict.pkl')


train_loader_all, test_loader_all = dataloader_10a_m2sfe(path=dataset_root,
                                                         batch_size=400, train_ratio=0.8)


net = torch.load(checkpoint_root)
net.eval()
correct = 0
fals = 0
cmt = torch.zeros(11, 11, 20, dtype=torch.int16)
for i, data in enumerate(test_loader_all):
    testdata, testlabel, SNR = data
    testdata, testlabel = testdata.cuda(), testlabel.cuda()
    outputs, _ = net(testdata)
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
    print(cmt[:, :, j])
    for k in range(11):
        num = num + cmt[k, k, j]
    print(int(num) / int(sum(sum(cmt[:, :, j]))))
