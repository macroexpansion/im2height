import torch
import torch.nn as nn
import numpy as np



def get_nonzero_value(x, y) -> 'x, y 4-dimensions':
    indices = (x > 0).nonzero()
    indexed_1 = x[indices[:,0], indices[:,1], indices[:,2], indices[:,3]]
    indexed_2 = y[indices[:,0], indices[:,1], indices[:,2], indices[:,3]]
    return indexed_1, indexed_2

l1 = nn.L1Loss()
t1 = torch.ones(1,1,3,3)
t2 = torch.zeros(1,1,3,3)
t3 = torch.ones(1,1,3,3)

y = torch.zeros(1,3,3,3)
x = torch.cat((t1,t2, t3)).reshape(1,3,3,3)

indexed_1, indexed_2 =  get_nonzero_value(y, x)
print(indexed_1 == torch.tensor([]))
print(indexed_2.size(0) == 0)

print(l1(indexed_1, indexed_2))



