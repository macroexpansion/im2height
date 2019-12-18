import torch
import torch.nn as nn


t1 = torch.rand(2,1,256,256)
t2 = torch.rand(2,1,256,256)
loss = nn.L1Loss()
l = loss(t1, t2)
print(l)