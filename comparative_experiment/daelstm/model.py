import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        outputs = []
        for t in range(x.size(1)):
            xt = x[:, t, :]
            output = self.layer(xt)
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class Dae6(nn.Module):
    def __init__(self):
      super(Dae6,self).__init__()
      # input.shape  = 128*2
      self.rnn1 = nn.LSTM(input_size=2,hidden_size=32,batch_first=True,dropout=0.6)
      self.rnn2 = nn.LSTM(input_size=32,hidden_size=32,batch_first=True,dropout=0.6)
      
      self.time = TimeDistributed(nn.Linear(32,2))

      self.fc = nn.Sequential(
          nn.Linear(in_features=32, out_features=32),
          nn.ReLU(),
          nn.BatchNorm1d(32),
          nn.Linear(in_features=32, out_features=16),
          nn.ReLU(),
          nn.BatchNorm1d(16),
          nn.Linear(in_features=16, out_features=11),
        )
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
      out1,(h1,c1) = self.rnn1(x)
      out2,(h2,c2) = self.rnn2(out1)
      logits = self.fc(h2[0])
      decoder = self.time(out2)    
      return logits,decoder,h2[0]



class Dae_w_fm(nn.Module):
    def __init__(self):
      super(Dae_w_fm,self).__init__()
      # input.shape  = 128*2
      self.rnn1 = nn.LSTM(input_size=2,hidden_size=32,batch_first=True,dropout=0.6)
      self.rnn2 = nn.LSTM(input_size=32,hidden_size=32,batch_first=True,dropout=0.6)
      
      self.time = TimeDistributed(nn.Linear(32,2))

      self.fc = nn.Sequential(
          nn.Linear(in_features=32, out_features=32),
          nn.ReLU(),
          nn.BatchNorm1d(32),
          nn.Linear(in_features=32, out_features=16),
          nn.ReLU(),
          nn.BatchNorm1d(16),
          nn.Linear(in_features=16, out_features=11),
        )
      self.feature_mapping = nn.Sequential(
         nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,padding=1,stride=1),
         nn.BatchNorm1d(16),
         nn.LeakyReLU(),
         nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding=1,stride=1),
         nn.BatchNorm1d(32),
         nn.LeakyReLU(),
         nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3,padding=1,stride=1),
         nn.BatchNorm1d(1),
         nn.LeakyReLU(),         
      )
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
      out1,(h1,c1) = self.rnn1(x)
      out2,(h2,c2) = self.rnn2(out1)    
      modfeat = self.feature_mapping(h2[0].unsqueeze(1))
      logits = self.fc(modfeat[:,0])
      decoder = self.time(out2)    
      return logits,decoder,h2[0]


if __name__=='__main__':

  from thop import profile
  from thop import clever_format

  model = Dae_w_fm()

  myinput = torch.zeros((1, 128, 2)).cuda()
  flops, params = profile(model.cuda(), inputs=(myinput,))
  flops, params = clever_format([flops, params], "%.3f")
  print('flops',flops)
  print('params',params)