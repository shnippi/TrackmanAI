import torch

a = torch.rand(1,2,3,4)
print(a.transpose(0,3).transpose(1,2).size())
print(a.permute(3,2,1,0).size())