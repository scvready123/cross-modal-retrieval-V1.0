
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import numpy as np



a = torch.Tensor([[1,2]]).cuda()

print(a.device)



