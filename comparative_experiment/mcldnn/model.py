import torch
import torch.nn as nn


class Mcldnn(nn.Module):
    def __init__(self):
      super(Mcldnn,self).__init__()
      # input1.shape  = 1,2,128
      # input1.shape  = 128,1

      dr = 0.5
      self.conv1 = nn.Sequential(
         nn.ZeroPad2d((3,4,1,0)),
         nn.Conv2d(in_channels=1,out_channels=50,kernel_size=(2,8)),
         nn.ReLU(),
      )

      self.conv2 = nn.Sequential(
         nn.ZeroPad2d((7,0,0,0)),
         nn.Conv1d(in_channels=1,out_channels=50,kernel_size=(8)),
         nn.ReLU(),
      )
      self.conv3 = nn.Sequential(
         nn.ZeroPad2d((7,0,0,0)),
         nn.Conv1d(in_channels=1,out_channels=50,kernel_size=(8)),
         nn.ReLU(),
      )
      self.conv4 = nn.Sequential(
         nn.ZeroPad2d((3,4,0,0)),
         nn.Conv2d(in_channels=50,out_channels=50,kernel_size=(1,8)),
         nn.ReLU(),
      )
      self.conv5 = nn.Sequential(
         nn.Conv2d(in_channels=100,out_channels=100,kernel_size=(2,5)),
         nn.ReLU(),
      )

     

      self.Rnn = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, batch_first=True)
      
      self.fc = nn.Sequential(
          nn.Linear(in_features=128, out_features=128),
          nn.Dropout(dr),
          nn.SELU(),
          nn.Linear(in_features=128, out_features=128),
          nn.Dropout(dr),
          nn.SELU(),
          nn.Linear(in_features=128, out_features=11),
         #  nn.Softmax(dim=1),
        )
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x,y,z):

      out_conv1 = self.conv1(x)
      out_conv2 = self.conv2(y)
      out_conv3 = self.conv3(z)
      out_conv2,out_conv3 = out_conv2.unsqueeze(2),out_conv3.unsqueeze(2)
      concat = torch.cat((out_conv2,out_conv3),dim=2)
      out_conv4 = self.conv4(concat)
      concat = torch.cat((out_conv1,out_conv4),dim=1)
      out_conv5 = self.conv5(concat)
      out_conv5 = out_conv5.contiguous().view(out_conv5.size(0),out_conv5.size(1),-1).transpose(2,1)
      out,(ht,ct) = self.Rnn(out_conv5)
      logits = self.fc(out[:,-1,:])
      return logits,out[:,-1,:]

if __name__=='__main__':

  from thop import profile
  from thop import clever_format

  model = Mcldnn()

  myinput1 = torch.zeros((1,1, 2, 128)).cuda()
  myinput2 = torch.zeros((1,1,128)).cuda()
  flops, params = profile(model.cuda(), inputs=(myinput1,myinput2,myinput2))
  flops, params = clever_format([flops, params], "%.3f")
  print('flops',flops)
  print('params',params)