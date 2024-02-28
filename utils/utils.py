import torch
import numpy as np

def mask(data):
    mask=torch.ones(data.shape[0],2,128)
    index=4*np.round(31*np.random.rand(data.shape[0],12))
    for m in range(data.shape[0]):
        mask[m,:,index[m,:]]=0
        mask[m,:,index[m,:]+1]=0
        mask[m,:,index[m,:]+2]=0
        mask[m,:,index[m,:]+3]=0          
    feed_data = torch.mul(data,mask)
    return feed_data
