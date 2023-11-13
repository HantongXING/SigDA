import torch
import torch.nn as nn


class DomainClassifer(nn.Module):
    def __init__(self):
      super(DomainClassifer,self).__init__()
      
      self.classifer = nn.Sequential(
          nn.Linear(in_features=6400, out_features=2048),
          nn.ReLU(),
          nn.Linear(in_features=2048, out_features=512),
          nn.ReLU(),
          nn.Linear(in_features=512, out_features=64),
          nn.ReLU(),
          nn.Linear(in_features=64, out_features=2))
      
      for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
      
      return self.classifer(x)



if __name__=='__main__':

  model = DomainClassifer()
  print(model)
  input = torch.randn(32, 6400)
  out= model(input)
  
  print(out.shape)
