#### This is the 3rd model. This is an implementation of 
#### the denoising of feature maps module from FAIR(Kaiming He)
#### paper.(Feature Denoising for Improving Adversarial Robustness)
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from DataLoader import *
import time

def weight_init(m):
	if isinstance(m, nn.Conv2d):
		torch.nn.init.xavier_uniform_(m.weight)
	elif isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)

class Maxout(nn.Module):
	def __init__(self, device, pool_size = 4):
		super().__init__()
		self._pool_size = pool_size
		self.device = device
		#print("Maxout", device)

	def forward(self, x):
		assert x.shape[1] % self._pool_size == 0, \
			'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
		m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
		return torch.squeeze(m, dim = 2).to(self.device)

class DenoisingModule(nn.Module):
	"""docstring for DenoisingModule"""
	def __init__(self, in_channels, out_channels, softmax, embed):
		super(DenoisingModule, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
		self.embed = embed
		self.softmax = softmax
		self.in_channels = in_channels
		self.out_channels = out_channels
		if self.embed:
			self.theta = nn.Conv2d(self.in_channels, self.out_channels, 1)
			self.phi = nn.Conv2d(self.in_channels, self.out_channels, 1)
		else:
			self.theta, self.phi = None, None

	def forward(self, x):
		in_channels, H, W = list(x.size())[1:]
		if self.embed:
			theta = self.theta(x)
			phi = self.phi(x)
			g = x
		else:
			theta = x
			phi = x
			g = x

		if self.in_channels > H * W or self.softmax:
			f = torch.einsum('niab,nicd->nabcd', [theta, phi])
			if softmax:
				orig_shape = f.size()
				f = f.view(-1, H * W, H * W)
				f = f / torch.sqrt(self.in_channels)
				f = F.softmax(f)
				f = torch.reshape(f, orig_shape)
			f = torch.einsum('nabcd,nicd->niab', [f, g])
		else:
			f = torch.einsum('nihw,njhw->nij', [phi, g])
			f = torch.einsum('nij,nihw->njhw', [f, theta])

		if not self.softmax:
			f = f / (H * W)

		f = torch.reshape(f, x.size())
		#print("Weights", self.conv1.weight.get_device())
		result = self.conv1(f)
		#print("DENOISE", result.get_device())
		return x + result

class ConvModel(nn.Module):
	"""docstring for ConvModel"""
	def __init__(self, device):
		super(ConvModel, self).__init__()
		#print(device)
		self.denoise1 = DenoisingModule(64, 64, softmax = False, embed = False).double().to(device)
		self.denoise2 = DenoisingModule(32, 32, softmax = False, embed = False).double().to(device)
		self.denoise3 = DenoisingModule(16, 16, softmax = False, embed = False).double().to(device)
		self.denoise4 = DenoisingModule(8, 8, softmax = False, embed = False).double().to(device)
		self.maxout = Maxout(device, 4)
		self.conv1 = nn.Conv2d(1, 256, kernel_size = (17, 3), padding = (8, 1))#Given/2
		self.conv2 = nn.Conv2d(64, 128, kernel_size = (33, 5), padding = (16, 2))
		self.conv3 = nn.Conv2d(32, 64, kernel_size = (65, 9), padding = (32, 4))
		self.conv4 = nn.Conv2d(16, 32, kernel_size = (129, 17), padding = (64, 8))
		self.conv5 = nn.Conv2d(8, 4, kernel_size = (1, 1), padding = 0)

	def forward(self, x):
		output = self.conv1(x)
		output = self.maxout(output)
		output = self.denoise1(output)
		#print(output.size())

		output = self.conv2(output)
		output = self.maxout(output)
		output = self.denoise2(output)
		#print(output.size())

		output = self.conv3(output)
		output = self.maxout(output)
		output = self.denoise3(output)
		#print(output.size())

		output = self.conv4(output)
		output = self.maxout(output)
		output = self.denoise4(output)
		#print(output.size())

		output = self.conv5(output)
		output = self.maxout(output)
		#print(output.size())
		
		return output

def train(model, loader, criterion, criterion2, optimizer, device, alpha = 0):
	model.train()
	for _, data in enumerate(loader):
		#print(data[1])
		inp = model(torch.unsqueeze(data[1], dim = 1).to(device)).to(device)
		out = data[0]
		loss = criterion(inp, torch.unsqueeze(out, dim = 1).to(device))
		#loss2 = criterion2(data[2].cuda(), data[-1].cuda())
		total_loss = loss #+ alpha * loss2
		loss = total_loss.mean()	
		loss.backward()
		print("LOSS - ", loss)
		optimizer.step()


def test(model, data):
	model.eval()
	out = model(data)
	d = {'CreatedRF' : torch.squeeze(out)}
	io.savemat('Created', d)


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ConvModel(device).double().to(device)
	model.apply(weight_init)
	model = nn.DataParallel(model).cuda()
	#if torch.cuda.device_count() > 1:
		#model = nn.DataParallel(model)
	#model.to(device)
	loader = Loader()
	data_loader = DataLoader(loader, batch_size = 8)
	criterion = nn.MSELoss().to(device)
	criterion2 = nn.MSELoss().to(device)
	parameters = list(model.parameters())
	optimizer = torch.optim.Adam(parameters, lr = 0.001)
	start = time.time()
	for epoch in range(10):
		train(model, data_loader, criterion, criterion2, optimizer, device)
		ep = time.time()
		print("Time taken for one epoch is %d" %(ep - start))
	torch.save(model.state_dict(), "/gpfs/fs1/home/rdulam/Neural_Networks/Models/Checkpoints/checkpoint-3.pt")
	print("DONE")
	end = time.time()
	print("Time in seconds taken for 10 epochs - %d" %(end - start))
	#test(model, torch.ones(1, 1, 2048, 128).double().cuda())


if __name__ == '__main__':
	main()
