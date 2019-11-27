import torch 
import torch.nn as nn
from Model1 import *

ones = torch.ones(1, 1, 2048, 128).double().cuda()
model = CNN().double().cuda()
model.load_state_dict(torch.load("/gpfs/fs1/home/rdulam/Neural_Networks/Models/Checkpoints/checkpoint-1.pt"))
model.eval()
wow = model(ones)
print(wow.size())

