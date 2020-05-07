import torch
import numpy as np
ap = []
l = [1,2,3,2,1,3]
a = torch.tensor([[1, 0, 0], [1, 0, 0], [2, 3, 3],[2,3,4],[2,3,4]])

unique, inverse = torch.unique(a, sorted=True, return_inverse=True,dim = 0)
perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
inverse, perm = inverse.flip([0]), perm.flip([0])
perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

print(unique)
print(perm)
ap.append(l[perm])
print(ap)