import torch
import torch.nn as nn
import os
import numpy as np
import scipy.io as io
#from torchsummary import summary
#from sklearn.feature_extraction.image import extract_patches_2d
import torch.nn.functional as F
from DataLoader import *
import time
from torch.autograd import Variable

def weight_init(m):
	if isinstance(m, nn.Conv2d):
		torch.nn.init.xavier_uniform_(m.weight)
	elif isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)

class Maxout(nn.Module):
	def __init__(self, pool_size = 4):
		super().__init__()
		self._pool_size = pool_size

	def forward(self, x):
		assert x.shape[1] % self._pool_size == 0, \
			'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
		m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
		return torch.squeeze(m).cuda()

### Should I use an Encoder-Decoder Architecture for the Generator? I didn't. For now.

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self):
		super(Generator, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size = (17, 3), padding = (8, 1)) #Given/2
		self.conv2 = nn.Conv2d(32, 64, kernel_size = (33, 5), padding = (16, 2))
		self.conv3 = nn.Conv2d(64, 32, kernel_size = (65, 9), padding = (32, 4))
		self.conv4 = nn.Conv2d(32, 8, kernel_size = (129, 17), padding = (64, 8))
		self.conv5 = nn.Conv2d(8, 1, kernel_size = (1, 1), padding = 0)

	def forward(self, x):
		output = self.conv1(x)
		output = self.conv2(output)
		output = self.conv3(output)
		output = self.conv4(output)
		output = self.conv5(output)
		return output

class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self):
		super(Discriminator, self).__init__()

		models = [nn.Conv2d(1, 32, kernel_size = (32, 2), stride = (1, 1))]
		models += [nn.MaxPool2d(kernel_size = (8, 2), stride = (2, 2))]
		models += [nn.Conv2d(32, 64, kernel_size = (32, 2), stride = (1, 1))]
		models += [nn.MaxPool2d(kernel_size = (8, 2), stride = (2, 2))]
		models += [nn.Conv2d(64, 128, kernel_size = (32, 2), stride = (1, 1))]
		models += [nn.MaxPool2d(kernel_size = (8, 2), stride = (2, 2))]
		models += [nn.Conv2d(128, 256, kernel_size = (32, 2), stride = (1, 1))]
		models += [nn.MaxPool2d(kernel_size = (8, 2), stride = (2, 2))]
		models += [nn.Conv2d(256, 512, kernel_size = (32, 2), stride = (1, 1))]
		models += [nn.MaxPool2d(kernel_size = (8, 2), stride = (2, 2))]
		models += [nn.Conv2d(512, 1024, kernel_size = (8, 2), stride = (2, 1))]
		models += [Maxout(4)]
		models += [nn.Conv2d(256, 512, kernel_size = (11, 2), stride = (1, 1))]
		models += [Maxout(4)]
		models += [nn.Linear(128, 32)]
		models += [nn.Linear(32, 1)]
		models += [nn.Sigmoid()]

		self.model = nn.Sequential(*models)

	def forward(self, x):
		#x = torch.cat((x, y), dim = 1)
		#print(x.size())
		x = self.model(x)
		return torch.squeeze(x)

#model = Discriminator()
#model = Generator()
#summary(model.cuda(), (1, 2048, 128))
#z = np.zeros((2048, 128))
#patches = extract_patches_2d(z, (400, 8))
#print(patches.shape)

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 1:
		print(torch.cuda.device_count())
		modelG = Generator().double().to(device)
		modelD = Discriminator().double().to(device)
	else:
		modelG = Generator().double().cuda()
		modelD = Discriminator().double().cuda()
	
	modelG.apply(weight_init)
	modelD.apply(weight_init)
	loader = Loader()
	data_loader = DataLoader(loader, batch_size = 12, drop_last = True)
	#device = torch.device("cuda")
	criterion = nn.BCELoss().to(device)
	criterion2 = nn.MSELoss().to(device)
	#parameters = list(model.parameters())
	optimizerD = torch.optim.Adam(modelD.parameters()) # lr is the Learning Rate.
	optimizerG = torch.optim.Adam(modelG.parameters())
	start = time.time()
	for epoch in range(8):
		print("TESTING!!")
		train(modelG, modelD, device, criterion, criterion2, optimizerG, optimizerD, data_loader, alpha = 1)
		print("Epoch = %d" %(epoch))
	torch.save(modelG.state_dict(), "/gpfs/fs1/home/rdulam/Neural_Networks/Models/Checkpoints/checkpoint-5-G.pt")
	torch.save(modelD.state_dict(), "/gpfs/fs1/home/rdulam/Neural_Networks/Models/Checkpoints/checkpoint-5-D.pt")
	print("DONE")
	end = time.time()
	print("Time in seconds taken for 10 epochs - %d" %(end - start))
	#test(model, torch.ones(1, 1, 2048, 128).double().cuda())

def train(modelG, modelD, device, criterion, criterion2, optimizerG, optimizerD, loader, alpha):
	

	modelG.train()
	modelD.train()

	for batch_idx, data in enumerate(loader):
		#print('Device No.', torch.cuda.current_device())
		#print(data[0].size())
		modelD.zero_grad()
		## Loss for the Discriminator.
		shape = list(data[0].size())
		#print(shape[0])
		batch_size = shape[0]
		label_real = Variable(torch.ones(batch_size).double().to(device))
		label_false = Variable(torch.zeros(batch_size).double().to(device))

		## Discriminator Real and Fake here itself.
		inp = modelG(torch.unsqueeze(data[1], dim = 1).to(device)).to(device)
		out = torch.unsqueeze(data[0], dim = 1).to(device)
		#noise = torch.squeeze(noise)
		input_D_R = modelD(out)
		input_D_F = modelD(inp)

		#input_D_R = torch.squeeze(input_D_R)
		#input_D_F = torch.squeeze(input_D_F)

		D_real_loss = criterion(input_D_R, label_real)
		D_real_loss.backward(retain_graph = True)

		D_fake_loss = criterion(input_D_F, label_false)
		D_fake_loss.backward(retain_graph = True)

		D_loss = D_real_loss + D_fake_loss
		optimizerD.step()
		#print("Discriminator Loss -", D_loss)

		modelG.zero_grad()
	
		err = torch.mean(torch.abs(inp - out).to(device)).to(device) # L-1 loss is being used.
		#err_bmode = torch.mean(torch.abs(inp - out).cuda()).cuda() Look for an efficient implementation of this method. 
		G_loss = criterion(input_D_F, label_real)
		G_total_loss = torch.mean(G_loss).to(device) + alpha * err 
		G_total_loss.backward()
		#print("Generator Loss -", G_total_loss)

		optimizerG.step()

if __name__ == '__main__':
	main()
		
