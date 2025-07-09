import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from . import utils_model
from matplotlib import rcParams as mpl_param
from .network_unet import *


class ImageDenoiserDRUNet():
	def __init__(self, model_path, color_mode='gray', device='cpu'):
		
		if color_mode == 'color':
			n_channels = 3                   # 3 for color image
		else:
			n_channels = 1                   # 1 for grayscale image
			
		# create models
		self.model = UNetRes(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        
		# Load saved weights
		self.model.load_state_dict(torch.load(model_path), strict=True)
		self.model.eval()
		for k, v in self.model.named_parameters():
			v.requires_grad = False
		self.model = self.model.to(device)
        
		self.device = device
		
        
    # perform denoising with DRUNet
	def inference(self, img, sig):
		
		img = img.to(self.device)
		img = img.permute(2, 0, 1).float().unsqueeze(0)
		img = torch.cat((img, torch.FloatTensor([sig]).repeat(1, 1, img.shape[2], img.shape[3]).to(self.device) ), dim=1)
        
		if img.size(2)//8==0 and img.size(3)//8==0:
			img_deno = self.model(img)
		else:
			img_deno = utils_model.test_mode(self.model, img, refield=64, mode=5)

		return img_deno

