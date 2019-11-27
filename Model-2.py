#### Trying out with Octave Convolutions. Add Activation, Dropout and Batch Normalization later.
#### Can I add the denoising module in model-3 to the low frequency feature maps of octave convolutions
#### in order to remove noise from the texture. Since, speckle is important here. If the noise is present 
#### in both high and low frequencies then the module should be added for both of them.


import torch
import torch.nn as nn
import os
import sys
import numpy as np

class OctaveCNN(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
				groups=1, bias=False):
		super(OctaveCNN, self).__init__()
		self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
		assert stride == 1 or stride == 2, "Stride should be 1 or 2."
		self.stride = stride
		assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
		self.alpha_in, self.alpha_out = alpha_in, alpha_out
		self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
						nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
								kernel_size, 1, padding, dilation, groups, bias)
		self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 else \
						nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
								kernel_size, 1, padding, dilation, groups, bias)
		self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 else \
						nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
								kernel_size, 1, padding, dilation, groups, bias)
		self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
						nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
								kernel_size, 1, padding, dilation, groups, bias)

	def forward(self, x):
		x_h, x_l = x if type(x) is tuple else (x, None)

		if x_h is not None:
			x_h = self.downsample(x_h) if self.stride == 2 else x_h
			x_h2h = self.conv_h2h(x_h)
			x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 else None
		if x_l is not None:
			x_l2h = self.conv_l2h(x_l)
			x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
			x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
			x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None 
			x_h = x_l2h + x_h2h
			x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
			return x_h, x_l
		else:
			return x_h2h, x_h2l

#### This Denoising module doesn't contain the 1 by 1 convolution and adding back the input through a residual connection.
#### It isn't present since adding back will bring back the noise. Maybe it'll also bring back the lost information during 
#### denoising. 

#### Should add the Denoising module in between the octave convolutions.
	
class ConvModel(nn.Module):
	"""docstring for ConvModel"""
	def __init__(self):
		super(ConvModel, self).__init__()
		self.conv1 = OctaveCNN(1, 16, 1, alpha_in = 0.0, alpha_out = 1.0)
		self.conv2 = OctaveCNN(16, 64, 1)
		self.conv3 = OctaveCNN(64, 32, 1)
		self.conv4 = OctaveCNN(32, 16, 1)
		self.conv5 = OctaveCNN(16, 4, 1)
		self.conv6 = OctaveCNN(4, 1, 1, alpha_in = 1.0 , alpha_out = 0.0)

	def forward(self, x):
		x_h, x_l = self.conv1(x)
		x_h, x_l = self.conv2((x_h, x_l))
		x_h, x_l = self.conv3((x_h, x_l))
		x_h, x_l = self.conv4((x_h, x_l))
		x_h, x_l = self.conv5((x_h, x_l))
		x_output = self.conv6((x_h, x_l))

		return x_output
		

def train():
	continue

def test():
	continue

def main():
	continue

if __name__ == 'main':
	main()
