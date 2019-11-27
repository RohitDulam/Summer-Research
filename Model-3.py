#### This is the 3rd model. This is an implementation of 
#### the denoising of feature maps module from FAIR(Kaiming He)
#### paper.(Feature Denoising for Improving Adversarial Robustness)
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class Maxout(nn.Module):
	def __init__(self, pool_size = 4):
		super().__init__()
		self._pool_size = pool_size

	def forward(self, x):
		assert x.shape[1] % self._pool_size == 0, \
			'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
		m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
		return torch.squeeze(m).cuda()

class DenoisingModule(nn.Module):
	"""docstring for DenoisingModule"""
	def __init__(self, in_channels, out_channels, softmax, embed):
		super(DenoisingModule, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, 1).cuda()
		self.embed = embed
		self.softmax = softmax
		self.in_channels = in_channels
		self.out_channels = out_channels
		if self.embed:
			self.theta = nn.Conv2d(self.in_channels, self.out_channels, 1).cuda()
			self.phi = nn.Conv2d(self.in_channels, self.out_channels, 1).cuda()
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
		result = self.conv1(f)

		return x + result

class ConvModel(nn.Module):
	"""docstring for ConvModel"""
	def __init__(self):
		super(ConvModel, self).__init__()

		self.denoise1 = DenoisingModule(64, 64, softmax = False, embed = False)
		self.denoise2 = DenoisingModule(32, 32, softmax = False, embed = False)
		self.denoise3 = DenoisingModule(16, 16, softmax = False, embed = False)
		self.denoise4 = DenoisingModule(8, 8, softmax = False, embed = False)
		self.maxout = Maxout(4)
		self.conv1 = nn.Conv2d(1, 256, kernel_size = (17, 3), padding = (8, 1)).cuda() #Given/2
		self.conv2 = nn.Conv2d(64, 128, kernel_size = (33, 5), padding = (16, 2)).cuda()
		self.conv3 = nn.Conv2d(32, 64, kernel_size = (65, 9), padding = (32, 4)).cuda()
		self.conv4 = nn.Conv2d(16, 32, kernel_size = (129, 17), padding = (64, 8)).cuda()
		self.conv5 = nn.Conv2d(8, 4, kernel_size = (1, 1), padding = 0).cuda()

	def forward(self, x):
		output = self.conv1(x)
		output = self.maxout(output)
		output = self.denoise1(output)
		print(output.size())

		output = self.conv2(output)
		output = self.maxout(output)
		output = self.denoise2(output)
		print(output.size())

		output = self.conv3(output)
		output = self.maxout(output)
		output = self.denoise3(output)
		print(output.size())

		output = self.conv4(output)
		output = self.maxout(output)
		output = self.denoise4(output)
		print(output.size())

		output = self.conv5(output)
		output = self.maxout(output)
		print(output.size())
		
		return output

def main():
	x = ConvModel()
	inp = torch.ones(2, 1, 2048, 128).cuda()
	output = x(inp)
	print("EXECUTED")

if __name__ == '__main__':
	main()
