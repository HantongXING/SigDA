import torch
import torch.nn as nn


class DNET(nn.Module):
    def __init__(self):
      super(DNET,self).__init__()

      self.encoder = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=50, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(50),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=50, out_channels=128, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(128),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(1024),
          nn.LeakyReLU())

      self.decoder = nn.Sequential(
          nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(128),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=128, out_channels=50, kernel_size=3, padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(50),
          nn.LeakyReLU(),
          nn.ConvTranspose1d(in_channels=50, out_channels=2, kernel_size=3,padding=1,output_padding=0,stride=1),
          nn.BatchNorm1d(2))

      self.conv = nn.Sequential(
              nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(512),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(256),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=256, out_channels=50, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(50),
              nn.LeakyReLU())
      
      self.Rnn = nn.LSTM(128, 128, num_layers=2, batch_first=True)
      
      self.fc = nn.Sequential(
          nn.Linear(in_features=6400, out_features=2048),
          nn.Dropout(0.6),
          nn.LeakyReLU(),
          nn.Linear(in_features=2048, out_features=1024),
          nn.Dropout(0.6),
          nn.LeakyReLU(),
          nn.Linear(in_features=1024, out_features=256),
          nn.Dropout(0.6 ),
          nn.LeakyReLU(),
          nn.Linear(in_features=256, out_features=11))
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
      x1=self.encoder(x)
      x2=self.decoder(x1)
      x1=self.conv(x1)
      x1,_=self.Rnn(x1)
      y = x1.contiguous().view(x1.size(0),-1)
      x = self.fc(y)
      return x, x2

if __name__=='__main__':

  model = DNET()
  print(model)
  input = torch.randn(32, 2, 128)
  out, inp = model(input)
  print(out.shape)
