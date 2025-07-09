#==============================================================================
# Model for the Video Denoising Network (ViDNet). The code is based on the 
# implementation of DRUNet ( https://github.com/cszn/DPIR ) and FastDVDnet 
# ( https://github.com/m-tassano/fastdvdnet ).
# 
# Author:   Yunhui Gao
# Date:     2025/06/10
#==============================================================================


import torch
import torch.nn as nn
from . import basicblock as B
import numpy as np
import math


class UNetRes(nn.Module):
    
    def __init__(self, io_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = B.conv((io_nc+1)*3, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.cupsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], io_nc, bias=False, mode='C')
        self.reset_params()
        
        
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            
            # pytorch default initializer
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)
            

    def forward(self, in1, in2, in3, noise_map):
        
        x1 = self.m_head(torch.cat((in1, noise_map, in2, noise_map, in3, noise_map), dim=1))
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x  = self.m_body(x4)
        x  = self.m_up3(x+x4)
        x  = self.m_up2(x+x3)
        x  = self.m_up1(x+x2)
        x  = self.m_tail(x+x1)

        return x


class VidNet(nn.Module):
    
    def __init__(self, io_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
       
        super(VidNet, self).__init__()
        
        # define two denoising blocks
        self.DenBlock1 = UNetRes(io_nc, nc, nb, act_mode, downsample_mode, upsample_mode)
        self.DenBlock2 = UNetRes(io_nc, nc, nb, act_mode, downsample_mode, upsample_mode)
        self.io_nc = io_nc
        
    
    def forward(self, x, noise_map):
        
        (x1, x2, x3, x4, x5) = tuple(x[:,i*self.io_nc:(i+1)*self.io_nc,:,:] for i in range(5))
        
        # stage one
        x21 = self.DenBlock1(x1, x2, x3, noise_map)
        x22 = self.DenBlock1(x2, x3, x4, noise_map)
        x23 = self.DenBlock1(x3, x4, x5, noise_map)
        
        # stage two
        x = self.DenBlock2(x21, x22, x23, noise_map)
        
        return x