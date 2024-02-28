#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:19:42 2021

@author: ws
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import numpy as np

class M2SFE_W_SFPA(nn.Module):
    def __init__(self):
      super(M2SFE_W_SFPA,self).__init__()
      self.feature_extractor = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=50, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(50),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=50, out_channels=256, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1,stride=1),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3,padding=1,stride=1),
          nn.BatchNorm1d(1024),
          nn.LeakyReLU())

      self.reconstructor = nn.Sequential(
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


      self.cnn_mapping = nn.Sequential(
              nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(512),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(256),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(128),
              nn.LeakyReLU(),
              nn.Conv1d(in_channels=128, out_channels=50, kernel_size=3,padding=1,stride=1),
              nn.BatchNorm1d(50),
              nn.LeakyReLU())
      
      self.rnn_mapping = nn.LSTM(126, 126, num_layers=2, batch_first=True)
      
      self.classifer = nn.Sequential(
          nn.Linear(in_features=6300, out_features=2048),
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
      bs,channel,dim=x.shape
      x1=self.feature_extractor(x)
      pool1 = F.avg_pool1d(x1,kernel_size=int(dim/2))
      pool2 = F.avg_pool1d(x1,kernel_size=int(dim/4))
      pool3 = F.avg_pool1d(x1,kernel_size=int(dim/8))
      pool4 = F.avg_pool1d(x1,kernel_size=int(dim/16))
      pool5 = F.avg_pool1d(x1,kernel_size=int(dim/32))
      pool6 = F.avg_pool1d(x1,kernel_size=int(dim/64))
      x = torch.cat((pool1,pool2,pool3,pool4,pool5,pool6),2)
      cons_input=self.reconstructor(x1)
      cnn_feature = self.cnn_mapping(x)
      rnn_feature,_=self.rnn_mapping(x)
      rnn_feature = x.contiguous().view(x.size(0),-1)
      logits = self.classifer(rnn_feature)
      return logits, rnn_feature,cons_input
