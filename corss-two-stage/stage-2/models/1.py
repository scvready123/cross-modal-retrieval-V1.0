import torch
from SCAN import cosine_similarity,l2norm
a = torch.Tensor([[1,2,3],[4,5,6],[0,0,0],[1,10,100]])
print(a)
print(a.mean(0))
b = a.mean(0).unsqueeze(0).contiguous().repeat(64, 1, 1)
print(b.size())

a = torch.randn(64,36,1024).mean(1)
b = torch.randn(32,50,1024).mean(1)

print(l2norm(torch.ones(2,4),dim=1))

print(torch.einsum("id,jd->ij",a,b).size())



