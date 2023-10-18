import torch
import torch.nn as nn


class Cldnn(nn.Module):
    def __init__(self):
      super(Cldnn,self).__init__()
      # input.shape  = 1,2,128
      dr = 0.5
      self.conv1 = nn.Sequential(
         nn.Conv2d(in_channels=1,out_channels=50,kernel_size=(1,8),padding=(0,2)),
         nn.ReLU(),
         nn.Dropout(dr),
      )

      self.conv2 = nn.Sequential(
         nn.Conv2d(in_channels=50,out_channels=50,kernel_size=(1,8),padding=(0,2)),
         nn.ReLU(),
         nn.Dropout(dr),
      )

      self.conv3 = nn.Sequential(
         nn.Conv2d(in_channels=50,out_channels=50,kernel_size=(1,8),padding=(0,2)),
         nn.ReLU(),
         nn.Dropout(dr),
      )

      self.Rnn = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True,dropout=0.5)
      
      self.fc = nn.Sequential(
          nn.Linear(in_features=50, out_features=256),
          nn.Dropout(dr),
          nn.ReLU(),
          nn.Linear(in_features=256, out_features=11),
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
      concat = torch.cat((out_conv1,out_conv3),dim=3)
      concat = concat.contiguous().view(concat.size(0),concat.size(1),-1).transpose(2,1)
      out,_ = self.Rnn(concat)
      logits = self.fc(out[:,-1,:])
      return logits,out[:,-1,:]
    



if __name__=='__main__':

  from thop import profile
  from thop import clever_format

  model = Cldnn()

  myinput = torch.zeros((1,1, 2, 128)).cuda()
  flops, params = profile(model.cuda(), inputs=(myinput,))
  flops, params = clever_format([flops, params], "%.3f")
  print('flops',flops)
  print('params',params)