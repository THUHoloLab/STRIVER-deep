import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import torch
import torch.nn as nn
from .models import FastDVDnet
from .fastdvdnet import denoise_seq_fastdvdnet
from .utils import *
from matplotlib import rcParams as mpl_param


NUM_IN_FR_EXT = 5   # temporal size of patch
OUTIMGEXT = '.png'  # output images format


class VideoDenoiserFastDVDnet():
	
    def __init__(self, model_path, device):
		
		# Create models
        self.model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
        
        self.model_temp = self.model_temp.to(device)
        self.model_temp.load_state_dict(torch.load(model_path), strict=True)
		
		# Sets the model in evaluation mode (e.g. it removes BN)
        self.model_temp.eval()
		
        self.device = device
        
    # perform denoising with FastDVDnet
    def inference(self, vid, sig):
        with torch.no_grad():
            vid = vid.to(self.device)
            sig = torch.FloatTensor([sig]).to(self.device)
            vid_deno = denoise_seq_fastdvdnet(seq=vid, noise_std=sig,
									 temp_psz=NUM_IN_FR_EXT, model_temporal=self.model_temp)
        return vid_deno
	
	
def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
	"""Saves the denoised and noisy sequences under save_dir
	"""
	seq_len = seqnoisy.size()[0]
	for idx in range(seq_len):
		# build Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir,\
						('n{}_{}').format(sigmaval, idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{}').format(sigmaval, idx) + fext)
		else:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

		# save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name, noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)
	
