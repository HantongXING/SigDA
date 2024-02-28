import torch
import torch.nn as nn


class LSTM2(nn.Module):
    def __init__(self):
      super(LSTM2,self).__init__()
      # input.shape  = 128,2

      self.Rnn = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True,dropout=0.2)
      self.fc = nn.Sequential(
          nn.Linear(in_features=128, out_features=11),
        )
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
      for name, param in self.Rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=1)
    
    def forward(self, x):
      out,(h,c) = self.Rnn(x)
      cls = self.fc(h[1])
      return cls,h[1]
 


# if __name__=='__main__':

#   from thop import profile
#   from thop import clever_format

#   model = LSTM2()

#   myinput = torch.zeros((1,128,2)).cuda()
#   flops, params = profile(model.cuda(), inputs=(myinput,))
#   flops, params = clever_format([flops, params], "%.3f")
#   print('flops',flops)
#   print('params',params)