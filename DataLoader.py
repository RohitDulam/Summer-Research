import torch
import torch.nn as nn
import os
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
import matlab
import matlab.engine as engine

class Loader(Dataset):
	"""docstring for Loader"""
	def __init__(self, data_dir = 'removed'):
		super(Loader, self).__init__()
		self.data_dir = data_dir
		self.data_c = []
		self.data_n = []
		self.eng = matlab.engine.start_matlab()
		i = 0
		for subdirs, dirs, files in os.walk(self.data_dir):
			if i == 0:
				i = 1;continue
			#print(subdirs + '/' + 'RFC.mat')
			#print(dirs)
			self.data_c.append(subdirs + '/' + 'RFC.mat')
			self.data_n.append(subdirs + '/' + 'RF.mat')


	def __getitem__(self, i):
		try:
			self.eng.addpath(r'removed', nargout = 0)
			c_data = io.loadmat(self.data_c[i])
			n_data = io.loadmat(self.data_n[i])

			#eng = engine.start_matlab()
			x = matlab.double(n_data['hun'].tolist())
			x = self.eng.genBmode(x)
			x = np.array(x._data).reshape((2048, 128), order = 'F')
			x = (x - x.min()) / (x.max() - x.min())

			y = matlab.double(c_data['RF0'].tolist())
			y = self.eng.genBmode(y)
			y = np.array(y._data).reshape((2048, 128), order = 'F')
			y = (y - y.min()) / (y.max() - y.min())

			return [c_data['RF0'], n_data['hun'], x, y]
		except ValueError:
			print("VALUE!!")
			#print(io.loadmat(self.data_c[i]))
			#print(io.loadmat(self.data_n[i]))
			print("single", self.data_c[i])
			print(self.data_n[i])			
			#return None
		except KeyError:
			print("KEY!!")
			print("single", self.data_c[i])
			print(self.data_n[i])
		

	def __len__(self):
		return len(self.data_c)

#if __name__ == '__main__':
	#test = Loader()
	#data_loader = DataLoader(test, batch_size = 10)
	#for x,y in enumerate(data_loader):
		#print(y[-1].size())
		#break
