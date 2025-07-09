#==============================================================================
# Wrapper class for the Video Denoising Network (ViDNet). The code is based on 
# the implementation of FastDVDnet ( https://github.com/m-tassano/fastdvdnet ).
# 
# Author:   Yunhui Gao
# Date:     2025/06/10
#==============================================================================


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from .model import VidNet
from matplotlib import rcParams as mpl_param
from .utils_video import *


class VideoDenoiserViDNet():
	
    def __init__(self, model_path, device):
		
		# create model
        self.model_temp = VidNet(io_nc=1,
                                 nc=[90, 128, 256, 512],
                                 nb=4,
                                 act_mode='R',
                                 upsample_mode='convtranspose',
                                 downsample_mode='strideconv')
		
        self.model_temp = self.model_temp.to(device)
        self.model_temp.load_state_dict(torch.load(model_path), strict=True)
		
        # set the model in evaluation mode
        self.model_temp.eval()
        self.device = device
        
    
    # perform denoising with ViDNet
    def inference(self, vid, sig):
        with torch.no_grad():
            vid.to(self.device)
            sig = torch.FloatTensor([sig]).to(self.device)
            vid_deno = denoise_seq(seq=vid, noise_std=sig,
								   temp_psz=5, model_temporal=self.model_temp)
        return vid_deno
	
