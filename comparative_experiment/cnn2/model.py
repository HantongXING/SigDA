import torch
import torch.nn as nn


class CNN2(nn.Module):
    def __init__(self):
      super(CNN2,self).__init__()
      # input1.shape  = 1,2,128     

      dr = 0.5
      self.conv1 = nn.Sequential(
         nn.ZeroPad2d((3,4,1,0)),
         nn.Conv2d(in_channels=1,out_channels=256,kernel_size=(2,8)),
         nn.ReLU(),
         nn.MaxPool2d((1,2)),
         nn.Dropout(dr),
      )

      self.conv2 = nn.Sequential(
         nn.ZeroPad2d((3,4,1,0)),
         nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(2,8)),
         nn.ReLU(),
         nn.MaxPool2d((1,2)),
         nn.Dropout(dr),
      )
      self.conv3 = nn.Sequential(
         nn.ZeroPad2d((3,4,1,0)),
         nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(2,8)),
         nn.ReLU(),
         nn.MaxPool2d((1,2)),
         nn.Dropout(dr),
      )
      self.conv4 = nn.Sequential(
         nn.ZeroPad2d((3,4,1,0)),
         nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(2,8)),
         nn.ReLU(),
         nn.MaxPool2d((1,2)),
         nn.Dropout(dr),
      )
     
      self.fc = nn.Sequential(
          nn.Linear(in_features=1024, out_features=128),
          nn.ReLU(),
          nn.Linear(in_features=128, out_features=11),
        )
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):

      out_conv1 = self.conv1(x)
      out_conv2 = self.conv2(out_conv1)
      out_conv3 = self.conv3(out_conv2)
      out_conv4 = self.conv4(out_conv3)
      out_conv4 = out_conv4.contiguous().view(out_conv4.size(0),-1)

      logits = self.fc(out_conv4)
      return logits,out_conv4

if __name__=='__main__':

  from thop import profile
  from thop import clever_format

  model = CNN2()

  myinput1 = torch.zeros((1,1, 2, 128)).cuda()
  flops, params = profile(model.cuda(), inputs=(myinput1,))
  flops, params = clever_format([flops, params], "%.3f")
  print('flops',flops)
  print('params',params)