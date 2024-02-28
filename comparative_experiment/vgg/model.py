import torch
import torch.nn as nn



class Vgg2(nn.Module):
    def __init__(self):
      super(Vgg2,self).__init__()
      # input.shape  = 2,128
      dr = 0.2
      self.conv1 = nn.Sequential(
         nn.ZeroPad2d((3,4,0,0)),
         nn.Conv1d(in_channels=2,out_channels=64,kernel_size=8),
         nn.Dropout(dr),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size=2,stride=2),

         nn.ZeroPad2d((3,4,0,0)),
         nn.Conv1d(in_channels=64,out_channels=64,kernel_size=8),
         nn.Dropout(dr),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size=2,stride=2),

         nn.ZeroPad2d((3,4,0,0)),
         nn.Conv1d(in_channels=64,out_channels=64,kernel_size=8),
         nn.Dropout(dr),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size=2,stride=2),

         nn.ZeroPad2d((3,4,0,0)),
         nn.Conv1d(in_channels=64,out_channels=64,kernel_size=8),
         nn.Dropout(dr),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size=2,stride=2),

         nn.ZeroPad2d((3,4,0,0)),
         nn.Conv1d(in_channels=64,out_channels=64,kernel_size=8),
         nn.Dropout(dr),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size=2,stride=2),

         nn.ZeroPad2d((3,4,0,0)),
         nn.Conv1d(in_channels=64,out_channels=64,kernel_size=8),
         nn.Dropout(dr),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size=2,stride=2),

         nn.ZeroPad2d((3,4,0,0)),
         nn.Conv1d(in_channels=64,out_channels=64,kernel_size=8),
         nn.Dropout(dr),
         nn.ReLU(),
         nn.MaxPool1d(kernel_size=2,stride=2),
      )

      self.fc = nn.Sequential(
          nn.Linear(in_features=64, out_features=32),
          nn.SELU(),
          nn.Dropout(dr),
          nn.Linear(in_features=32, out_features=11),
        )
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
      out_conv1 = self.conv1(x)
      out_conv1 = out_conv1.contiguous().view(out_conv1.size(0),-1)
      logits = self.fc(out_conv1)
      return logits,out_conv1




if __name__=='__main__':

  from thop import profile
  from thop import clever_format

  model = Vgg2()

  myinput = torch.zeros((1, 2, 128)).cuda()
  flops, params = profile(model.cuda(), inputs=(myinput,))
  flops, params = clever_format([flops, params], "%.3f")
  print('flops',flops)
  print('params',params)