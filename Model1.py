#### This model is just a vanilla CNN without any Max-pooling 
#### layers. Just like Gasse et al. Also, trying to experiment 
#### with the different receptive fields. 

#### Rather than using maxout, I'll use squeeze and excite. 
#### Maybe this will make difference. Is this a good design
#### choice?

#### data i.e. the radio frequency images should be of dimensions
#### 1332 x 128. 

#### Double the receptive field since the input is 2048 x 128.


import torch
import torch.nn as nn
import os 
import numpy as np
from DataLoader import *
import scipy.io as io
import time

class Maxout(nn.Module):
	def __init__(self, pool_size = 4):
		super().__init__()
		self._pool_size = pool_size

	def forward(self, x):
		assert x.shape[1] % self._pool_size == 0, \
			'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
		m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
		return torch.squeeze(m, dim = 2).cuda()

class CNN(nn.Module):
	"""docstring for CNN"""
	def __init__(self):
		super(CNN, self).__init__()
		self.maxout = Maxout(4)
		self.conv1 = nn.Conv2d(1, 256, kernel_size = (17, 3), padding = (8, 1)).cuda() #Given/2
		self.conv2 = nn.Conv2d(64, 128, kernel_size = (33, 5), padding = (16, 2)).cuda()
		self.conv3 = nn.Conv2d(32, 64, kernel_size = (65, 9), padding = (32, 4)).cuda()
		self.conv4 = nn.Conv2d(16, 32, kernel_size = (129, 17), padding = (64, 8)).cuda()
		self.conv5 = nn.Conv2d(8, 4, kernel_size = (1, 1), padding = 0).cuda()

	def forward(self, x):
		#print(x.size())
		output = self.conv1(x.cuda())
		output = self.maxout(output)
		#print(output.size())

		output = self.conv2(output)
		output = self.maxout(output)
		#print(output.size())

		output = self.conv3(output)
		output = self.maxout(output)
		#print(output.size())

		output = self.conv4(output)
		output = self.maxout(output)
		#print(output.size())

		output = self.conv5(output)
		output = self.maxout(output)
		#print(output.size())

		return output

def train(model, loader, criterion, criterion2, optimizer, alpha = 0.1):
	model.train()
	for _, data in enumerate(loader):
		#print(data[1])
		inp = model(torch.unsqueeze(data[1], dim = 1)).cuda()
		out = data[0]
		loss = criterion(torch.squeeze(inp).cuda(), out.cuda())
		loss2 = criterion2(data[2].cuda(), data[-1].cuda())
		total_loss = loss + alpha * loss2
		total_loss.backward()
		print("LOSS - ", total_loss)
		optimizer.step()


def test(model, data):
	model.eval()
	out = model(data)
	d = {'CreatedRF' : torch.squeeze(out)}
	io.savemat('Created', d)
	

def main():
	model = CNN().double().cuda()
	loader = Loader()
	data_loader = DataLoader(loader, batch_size = 5)
	criterion = nn.MSELoss().cuda()
	criterion2 = nn.MSELoss().cuda()
	parameters = list(model.parameters())
	optimizer = torch.optim.Adam(parameters, lr = 0.001)
	start = time.time()
	for epoch in range(2):
		train(model, data_loader, criterion, criterion2, optimizer)
	#torch.save(model.state_dict(), "/gpfs/fs1/home/rdulam/Neural_Networks/Models/Checkpoints/checkpoint-1.pt")
	print("DONE")
	end = time.time()
	print("Time in seconds taken for 10 epochs - %d" %(end - start))
	#test(model, torch.ones(1, 1, 2048, 128).double().cuda())
	

if __name__ == '__main__':
	main()
