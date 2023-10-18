import torch
import torch.nn as nn

class Basenet(nn.Module):
    def __init__(self,in_filter,filter,max_pool):
      super(Basenet,self).__init__()
      self.max_pool = max_pool
      
      self.conv = nn.Conv2d(in_channels=in_filter,out_channels=filter,kernel_size=(1,1))
      self.conv1 = nn.Sequential(
         nn.ZeroPad2d((1,0,1,1)),
         nn.Conv2d(in_channels=filter,out_channels=filter,kernel_size=(3,2)),
         nn.BatchNorm2d(filter),
         nn.ReLU(),
         nn.ZeroPad2d((1,0,1,1)),
         nn.Conv2d(in_channels=filter,out_channels=filter,kernel_size=(3,2)),
         nn.BatchNorm2d(filter),
      )
      self.conv2 = nn.Sequential(
         nn.ZeroPad2d((1,0,1,1)),
         nn.Conv2d(in_channels=filter,out_channels=filter,kernel_size=(3,2)),
         nn.BatchNorm2d(filter),
         nn.ReLU(),
         nn.ZeroPad2d((1,0,1,1)),
         nn.Conv2d(in_channels=filter,out_channels=filter,kernel_size=(3,2)),
         nn.BatchNorm2d(filter),
      )
      self.relu = nn.ReLU()
      self.bn = nn.BatchNorm2d(32)
      self.pool = nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))

    def forward(self, x): 
       X_shortcut = x     
       # 第一层卷积
       x1 = self.conv(x)
       x1 = self.bn(x1)
       x1 = self.relu(x1)

       # 第一层残差
       x2 = self.conv1(x1)
       x2 = self.relu(x2 + X_shortcut) 

       # 第二层残差
       X_shortcut = x2
       x3 = self.conv2(x2)
       x3 = self.relu(x3 + X_shortcut)
       
       if self.max_pool:
          return self.pool(x3)
       else:
        return x3



class Resnet(nn.Module):
    def __init__(self):
      super(Resnet,self).__init__()
      # input.shape  = 1,128,2
      dr = 0.5
      self.layer1 = Basenet(in_filter=1,filter=32,max_pool=False)
      self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))

      self.layer2 = Basenet(in_filter=32,filter=32,max_pool=True)

      self.layer3 = Basenet(in_filter=32,filter=32,max_pool=True)

      self.layer4 = Basenet(in_filter=32,filter=32,max_pool=True)

      self.layer5 = Basenet(in_filter=32,filter=32,max_pool=True)

      self.fc = nn.Sequential(
         nn.Linear(in_features=128,out_features=64),
         nn.SELU(),
         nn.AlphaDropout(0.3),         
         nn.Linear(in_features=64,out_features=64),
         nn.SELU(),
         nn.AlphaDropout(0.3),
         nn.Linear(in_features=64,out_features=11),
      )
     

      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
      x1 = self.layer1(x)
      x1 = self.max_pool(x1)

      x2 = self.layer2(x1)
      x3 = self.layer2(x2)   # 32*16 *1
      x4 = self.layer3(x3)   # 32*8 *1
      x5 = self.layer4(x4)   # 32*4 *1

      x5 = x5.contiguous().view(x5.size(0),-1)
      out = self.fc(x5)

      return out, x5


if __name__=='__main__':

  from thop import profile
  from thop import clever_format

  model = Resnet()

  myinput = torch.zeros((1, 1, 128, 2)).cuda()
  flops, params = profile(model.cuda(), inputs=(myinput,))
  flops, params = clever_format([flops, params], "%.3f")
  print('flops',flops)
  print('params',params)