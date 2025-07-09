#==============================================================================
# This code provides a demonstration of video plug-and-play (PnP) algorithm for
# dynamic holographic reconstruction based on simulation.
# 
# Author:   Yunhui Gao
# Date:     2025/06/10
#==============================================================================


import math
import time
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import models.drunet.utils_image as util
import models.drunet.utils_model
from matplotlib import rcParams as mpl_param
from utils import *
from torchvision import transforms
from PIL import Image
from models.fastdvdnet.wrapper import *
from models.drunet.wrapper import *
from models.fastdvdnet.wrapper import *
from models.vidnet.wrapper import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from tabulate import tabulate


mpl_param["figure.dpi"] = 500


#=================================================
# define constants and auxiliary functions
#=================================================


# define constants
PI = math.pi        # math constant
DEVICE_INDEX = 0    # gpu index
SCALE = 1.2         # scaling factor


# define auxiliary functions
def transfunc_propagate(in_shape, dist, pxsize, wavlen):
    '''
    Calculate the transfer function for light propagation based on the angular 
    spectrum method.

    Parameters
    ----------
    in_shape : tuple of int
        2D dimension of the wavefront (pixel).
    dist : float
        Propagation distance (mm).
    pxsize : float
        Pixel size (mm).
    wavlen : float
        Wavelength (mm).

    Returns
    -------
    H: torch.Tensor
        Transfer function.
    '''
    # calculate parameters
    k = 2*PI/wavlen
    n1, n2 = in_shape[0], in_shape[1]
    
    # define meshgrid
    k1 = PI/pxsize*torch.linspace(-1, 1-2/n1, n1)
    k2 = PI/pxsize*torch.linspace(-1, 1-2/n2, n2)
    kk1, kk2 = torch.meshgrid(k1, k2, indexing='ij')
    
    # calculate transfer function
    H = torch.exp(1j*dist*torch.sqrt(k**2 - kk1**2 - kk2**2))
    
    return H


def transfunc_imshift(in_shape, shift):
    '''
    Calculate the transfer function for the lateral translation operator.

    Parameters
    ----------
    in_shape : tuple of int
        2D dimension of the input image (pixel).
    shift : 2-element array of float
        Translation distances (pixel) along the two axes.

    Returns
    -------
    H : torch.Tensor
        Transfer function.
    '''
    # define meshgrid
    n1, n2 = in_shape[0], in_shape[1]
    
    # define meshgrid
    f1 = torch.linspace(-n1/2,n1/2-1,n1)
    f2 = torch.linspace(-n2/2,n2/2-1,n2)
    u1, u2 = torch.meshgrid(f1, f2, indexing='ij')
    
    # calculate transfer function
    H = torch.exp(-1j*2*PI*(shift[0]*u1/n1 + shift[1]*u2/n2))
    
    return H


def D(x):
    '''
    Calculate the gradient (finite difference) field of a 3D datacube.
    
    Parameters
    ----------
    x : torch.Tensor
        3D spatiotemporal datacube.

    Returns
    -------
    g : torch.Tensor
        4D gradient tensor.
    '''
    # calculate gradient along each dimension
    g1 = x - torch.cat((x[1:,:,:], x[-1,None,:,:]), 0)
    g2 = x - torch.cat((x[:,1:,:], x[:,-1,None,:]), 1)
    g3 = x - torch.cat((x[:,:,1:], x[:,:,-1,None]), 2)
    
    # concatenate all gradients in the forth dimension
    g = torch.cat((g1[:,:,:,None], g2[:,:,:,None], g3[:,:,:,None]), 3)
    
    return g


def DT(g):
    '''
    Calculate the adjoint (transpose) operator of the gradient operator D.
    
    Parameters
    ----------
    g : torch.Tensor
        4D tensor.

    Returns
    -------
    u : torch.Tensor
        3D tensor.
    '''
    u1 = g[:,:,:,0] - torch.roll(g[:,:,:,0],1,0)
    u1[0,:,:] = g[0,:,:,0]
    u1[-1,:,:] = -g[-2,:,:,0]
    
    u2 = g[:,:,:,1] - torch.roll(g[:,:,:,1],1,1)
    u2[:,0,:] = g[:,0,:,1]
    u2[:,-1,:] = -g[:,-2,:,1]
    
    u3 = g[:,:,:,2] - torch.roll(g[:,:,:,2],1,2)
    u3[:,:,0] = g[:,:,0,2]
    u3[:,:,-1] = -g[:,:,-2,2]
    
    u = u1+u2+u3
    
    return u

    
def reg_param(lam_start, lam_end, iter_idx, n_iters, alpha=5):
    '''
    Calculate the regularization parameters at each iteration, with 
    exponentially decaying values.
    
    Parameters
    ----------
    lam_start : np.array
                Regularization parameters at the first iteration.
    lam_end   : np.array
                Asymptotic values for the regularization parameters.
    iter_idx  : int
                Current iteration number.
    n_iters   : int
                Total number of iterations.
    alpha     : float
                Hyperparameter controlling the decaying rate.
                
    Returns
    -------
    lam       : np.array
                Parameter values at current iteration.
    '''
    # define exponentially decaying values
    lam = lam_end + (lam_start-lam_end) * np.exp(-alpha*(iter_idx-1)/n_iters)
    
    return lam


def denoise_vidnet(x, model, sig, scale=1):
    '''
    Apply ViDNet to a complex-valued spatiotemporal datacube given a 
    prespecified noise level.
    
    Parameters
    ----------
    x     : torch.Tensor
            Complex-valued 3D spatiotemporal datacube.
    model : 
            Wrapper class for the pretrained ViDNet model.    
    sig   : float
            Noise level.
    scale : float
            Scaling parameter applied to the real and imaginary parts before 
            performing denoising.
                
    Returns
    -------
    y     : torch.Tensor
            Denoised complex-valued datacube.
    '''
    # convert shape
    x = torch.permute(x, (2,0,1))
    x = torch.unsqueeze(x, dim=1)
    
    # scale the value
    x_r = (torch.real(x) + scale) / scale / 2
    x_i = (torch.imag(x) + scale) / scale / 2
    
    # perform denoising to the real and imaginary parts separately
    y_r = model.inference(x_r, sig)
    y_i = model.inference(x_i, sig)
    
    # convert shape
    y_r = torch.permute(y_r[:,0,:,:], (1,2,0))
    y_i = torch.permute(y_i[:,0,:,:], (1,2,0))
    
    # rescale the value
    y_r = torch.squeeze(y_r) * scale * 2 - scale
    y_i = torch.squeeze(y_i) * scale * 2 - scale
    
    # calculate the complex datacube
    y = y_r + 1j*y_i
    
    return y


def denoise_fastdvdnet(x, model, sig, scale=1):
    '''
    Apply FastDVDnet to a complex-valued spatiotemporal datacube given a 
    prespecified noise level.
    
    Parameters
    ----------
    x     : torch.Tensor
            Complex-valued 3D spatiotemporal datacube.
    model : 
            Wrapper class for the pretrained FastDVDnet model.    
    sig   : float
            Noise level.
    scale : float
            Scaling parameter applied to the real and imaginary parts before 
            performing denoising.
    
    Returns
    -------
    y     : torch.Tensor
            Denoised complex-valued datacube.
    '''
    # convert shape
    x = torch.permute(x, (2,0,1))
    x = torch.unsqueeze(x, dim=1)
    
    # scale the value
    x_r = (torch.real(x) + scale) / scale / 2
    x_i = (torch.imag(x) + scale) / scale / 2
    
    # perform denoising to the real and imaginary parts separately
    y_r = model.inference(x_r, sig)
    y_i = model.inference(x_i, sig)
    
    # convert shape
    y_r = torch.permute(y_r[:,0,:,:], (1,2,0))
    y_i = torch.permute(y_i[:,0,:,:], (1,2,0))
    
    # rescale the value
    y_r = torch.squeeze(y_r) * scale * 2 - scale
    y_i = torch.squeeze(y_i) * scale * 2 - scale
    
    # calculate the complex datacube
    y = y_r + 1j*y_i
    
    return y


def denoise_drunet(x, model, sig, scale=1):
    '''
    Apply DRUNet to a complex-valued spatiotemporal datacube given a 
    prespecified noise level.
    
    Parameters
    ----------
    x     : torch.Tensor
            Complex-valued 3D spatiotemporal datacube.
    model : 
            Wrapper class for the pretrained DRUNet model.    
    sig   : float
            Noise level.
    scale : float
            Scaling parameter applied to the real and imaginary parts before 
            performing denoising.
                
    Returns
    -------
    y     : torch.Tensor
            Denoised complex-valued datacube.
    '''
    # calculate the sequence length
    K = x.shape[2]
    
    # scale the value
    x_r = (torch.real(x) + scale) / scale / 2
    x_i = (torch.imag(x) + scale) / scale / 2
    
    # perform frame-by-frame denoising
    y_r, y_i = [], []
    for k in range(K):
        
        u = torch.unsqueeze(x_r[:,:,k], dim=2)
        u = model.inference(u, sig)
        u = torch.squeeze(u)
        u = torch.unsqueeze(u, dim=2)
        y_r.append(u)
        
        u = torch.unsqueeze(x_i[:,:,k], dim=2)
        u = model.inference(u, sig)
        u = torch.squeeze(u)
        u = torch.unsqueeze(u, dim=2)
        y_i.append(u)
        
    y_r = torch.cat(y_r, dim=2)
    y_i = torch.cat(y_i, dim=2)
    
    # rescale the value
    y_r = y_r * scale * 2 - scale
    y_i = y_i * scale * 2 - scale
    
    # calculate the complex datacube
    y = y_r + 1j*y_i

    return y


def padimg_2(img, bias, size_out):
    '''
    Pad an image with zeros.
    
    Parameters
    ----------
    img      : torch.Tensor
               Input 2D image.
    bias     : tuple of int
               Padding size before the original image for the two dimensions.
    size_out : tuple of int
               Dimension of the output image.
                
    Returns
    -------
    img_out  : torch.Tensor
               Zero-padded image.
    '''
    img_out = torch.zeros(size_out)
    nw, nh  = img.shape
    img_out[bias[0]:bias[0]+nw,bias[1]:bias[1]+nh] = img
    
    return img_out


def cropimg_2(img, rect):
    '''
    Crop an image given a crop rectangle.
    
    Parameters
    ----------
    img  : torch.Tensor
           Input image.        
    rect : tuple of int
           Size and position of the crop rectangle, which is a 4-element tuple
           of the form (xmin, ymin, width, height).
           
    Returns
    -------
    img_out  : torch.Tensor
               Cropped image.
    '''
    img_out = img[rect[1]-1:rect[1]+rect[3]-1, rect[0]-1:rect[0]+rect[2]-1]
    
    return img_out


#=================================================
# main function
#=================================================


if __name__ == "__main__":
    
    save_results = False
    
    #=================================================
    # define sample filed
    #=================================================
    
    '''
    clipnames:  'akiyo','bus','coastguard','container','flower','football',
                'garden','hall_monitor','ice','news','salesman','stefan',
                'suzie','waterfall'
    '''
    # select test clip
    clipname = 'akiyo'
 
    # number of diversity measurements
    K = 20
    
    # define simulation parameters
    MASK_FEAT_SIZE = 2  # feature size (pixel) of the diffuser
    PXSIZE = 5e-3       # pixel size (mm)
    WAVLEN = 5e-4       # wavelength (mm)
    DIST_1 = 5          # sample-to-diffuser distance (mm)
    DIST_2 = 5          # diffuser-to-sensor distance (mm)
    
    # define padding parameters
    padsize_1 = 20
    padsize_2 = 20
    
    # define translation parameters
    shift_step = MASK_FEAT_SIZE
    shifts = np.array([np.linspace(0,(K-1)*shift_step,K),
                       np.linspace(0,0,K)]).astype(np.int16)
    shift_range_1 = np.max(np.abs(shifts[0,:]))
    shift_range_2 = np.max(np.abs(shifts[1,:]))
    
    # set random seed for reproducibility
    torch.manual_seed(0)
    
    PATH = './data/sim/' + clipname + '/QCIF'
    vid = []
    seq, _, _ = open_sequence(seq_dir=PATH, gray_mode=False,
                            expand_if_needed=False, max_num_fr=300)
    seq = seq[0:0+K,0,:,:].transpose(1,2,0)
    seq = torch.tensor(seq)
    seq = seq[0:256,0:256,:]
    Mh, Mw = seq.shape[0], seq.shape[1]
    
    # define the diffuser profile
    mask_size_1 = Mh+2*padsize_2
    mask_size_2 = Mw+2*padsize_2
    mask = torch.bernoulli(0.5*torch.ones(round((Mh+2*padsize_2+shift_range_1)/MASK_FEAT_SIZE), round((Mw+2*padsize_2+shift_range_2)/MASK_FEAT_SIZE)))
    mask_transform = transforms.Resize([Mh+2*padsize_2+shift_range_1,Mw+2*padsize_2+shift_range_2],interpolation=transforms.InterpolationMode.NEAREST)
    mask = mask_transform(mask[None,:])
    mask = mask[0,:]
    
    # load the test video clip
    fr_start = 0    # starting frame index
    fr_speed = 1    # frame interval
    vid = []
    seq, _, _ = open_sequence(seq_dir=PATH, gray_mode=False,
                            expand_if_needed=False, max_num_fr=300)
    if fr_speed == 0:
        seq = seq[:,fr_start,:,:].transpose(1,2,0)
        seq = torch.unsqueeze(torch.tensor(seq[:,:,0]),dim=2)
        seq = [seq for _ in range(K)]
        seq = torch.cat(seq,dim=2)
    else:
        seq = seq[fr_start:fr_start+fr_speed*K:fr_speed,0,:,:].transpose(1,2,0)
        seq = torch.tensor(seq)
    
    seq = seq[0:256,0:256,:]
    
    Mh, Mw = seq.shape[0], seq.shape[1]
    vid = []
    for k in range(K):
        img = padimg(seq[:,:,k], padsize_1+padsize_2)
        img = torch.tensor(np.pad(seq[:,:,k], padsize_1+padsize_2, mode='edge'))
        vid.append(torch.unsqueeze(img,dim=2))
    vid = torch.cat(vid, dim=2)
    
    # define translated diffuser profiles for each measurement
    masks = []
    for k in range(K):
        masks.append(torch.unsqueeze(mask[shifts[0,fr_start+k]:shifts[0,fr_start+k]+mask_size_1,shifts[1,fr_start+k]:shifts[1,fr_start+k]+mask_size_2],dim=2))
    masks = torch.cat(masks,dim=2)
    
    # define sample transmission function
    x = (0.5+0.5*vid) * torch.exp(1j*vid*PI/2)
    
    
    #=================================================
    # define forward model
    #=================================================
    
    torch.cuda.empty_cache()
    
    Nh = Mh + 2*padsize_2 + 2*padsize_1
    Nw = Mw + 2*padsize_2 + 2*padsize_1
    
    # calculate the transfer functions
    HQ1 = torch.fft.fftshift(transfunc_propagate((Nh, Nw), DIST_1, PXSIZE, WAVLEN))
    HQ2 = torch.fft.fftshift(transfunc_propagate((Mh+2*padsize_2, Mw+2*padsize_2), DIST_2, PXSIZE, WAVLEN))
    
    Q1  = lambda x: torch.fft.ifft2(torch.fft.fft2(x) * HQ1)                # free-space propagation from sample to diffuser
    Q1H = lambda x: torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(HQ1))    # Hermitian operator of Q1
    C1  = lambda x: cropimg(x, padsize_1)                                   # image cropping
    C1T = lambda x: padimg(x, padsize_1)                                    # transpose (Hermitian) operator of C1
    M   = lambda x, k: x * masks[:,:,k]                                     # diffuser modulation
    MH  = lambda x, k: x * torch.conj(masks[:,:,k])                         # Hermitian operator of M
    Q2  = lambda x: torch.fft.ifft2(torch.fft.fft2(x) * HQ2)                # free-space propagation from diffuser to sensor
    Q2H = lambda x: torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(HQ2))    # Hermitian operator of Q2
    C2  = lambda x: cropimg(x, padsize_2)                                   # image cropping
    C2T = lambda x: padimg(x, padsize_2)                                    # transpose (Hermitian) operator of C2
    A   = lambda x, k: C2(Q2(M(C1(Q1(x)),k)))                               # overall measurement operator
    AH  = lambda x, k: Q1H(C1T(MH(Q2H(C2T(x)),k)))                          # Hermitian operator of A
    
    # define measurement noise
    noise_level = 0.03
    
    # simulate measurement
    y = []
    for k in range(K):
        meas = torch.abs(A(x[:,:,k],k))**2
        meas = torch.maximum(meas + torch.randn(meas.shape)*noise_level, torch.tensor([0]))
        meas = torch.unsqueeze(meas, dim=2)
        y.append(meas)
    y = torch.cat(y, dim=2)
    
    # save simulation settings as a .mat file
    if save_results:
        o_filename = 'cache_sim/settings_'+clipname+'_'+str(fr_start)+'_'+'{:.3f}'.format(noise_level)+'.mat'
        scipy.io.savemat(o_filename,{'x': x.numpy(), 'y': y.numpy(), 'mask': mask.numpy(), 'N1': Nh, 'N2': Nw, 
                                     'M1': Mh, 'M2': Mw, 'K': K, 'PXSIZE': PXSIZE, 'WAVLEN': WAVLEN, 
                                     'DIST_1': DIST_1, 'DIST_2': DIST_2, 'shifts': shifts,
                                     'padsize_1': padsize_1, 'padsize_2': padsize_2})
    
    #=================================================
    # reconstruction algorithm (denoiser: TV)
    #=================================================
    
    K_recon = K     # number of reconstructed time points
    KK = lambda x: np.floor(K_recon*k/K).astype(int)    # round to integer indices
    
    # initial estimate
    x_est = torch.ones(Nh,Nw,K_recon).type(torch.complex64)
    
    device = torch.device("cuda:{}".format(DEVICE_INDEX) if torch.cuda.is_available() else "cpu")
    
    # define the denoising operators in sequence
    denoiser_seq = 'tv'
    denoisers = denoiser_seq.split('+')
    
    # define the regularization parameters
    lams = []
    for denoiser in denoisers:
        
        # TV denoiser
        if denoiser.lower() == 'tv':
            
            def lam(x):
                lam_start = np.array([1e-1, 1e-1, 1e-1])    # three-element array for x, y, and t regularization
                lam_end   = np.array([1e-3 + 0.1*noise_level, 1e-3 + 0.1*noise_level, 1e-3 + 0.1*noise_level])
                return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
            lams.append(lam)
         
    # algorithm parameters
    gam = 2         # step size
    n_iters = 200   # number of iterations
    n_subiters = 5  # number of subiterations for the TV-based proximal update
    
    # define auxiliary variables
    z_est = x_est
    g_est = torch.zeros(x_est.shape).type(torch.complex64)
    v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
    w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
    
    x_est     = x_est.to(device)
    y         = y.to(device)
    masks     = masks.to(device)
    HQ1       = HQ1.to(device)
    HQ2       = HQ2.to(device)
    g_est     = g_est.to(device)
    z_est     = z_est.to(device)
    v_est     = v_est.to(device)
    w_est     = w_est.to(device)
    
    time_start = time.time()
    
    verbose = False
    display = False
    display_iter = 1
    
    # main loop
    for i in range(n_iters):
        
        if verbose:
            print('Iter {:0>4d} / {:0>4d}'.format(i+1, n_iters))
        
        # gradient update
        g_est[:] = 0
        for k in range(K):
            u = A(z_est[:,:,KK(k)],k)
            u = (torch.abs(u) - torch.sqrt(y[:,:,k])) * torch.exp(1j*torch.angle(u))
            g_est[:,:,KK(k)] = g_est[:,:,KK(k)] + 1/2/(K/K_recon) * AH(u,k)
        
        u = z_est - gam * g_est
        
        # proximal update
        for idx, denoiser in enumerate(denoisers):
            lam = lams[idx](i)
            if denoiser == 'tv':
                lam = torch.Tensor(lam).to(device)
                v_est[:] = 0
                w_est[:] = 0
                for j in range(n_subiters):
                    w_next = v_est + 1/12/gam * D(u-gam*DT(v_est))
                    w_next[:,:,:,0] = torch.minimum(torch.abs(w_next[:,:,:,0]), lam[0]) * torch.exp(1j*torch.angle(w_next[:,:,:,0]))
                    w_next[:,:,:,1] = torch.minimum(torch.abs(w_next[:,:,:,1]), lam[1]) * torch.exp(1j*torch.angle(w_next[:,:,:,1]))
                    w_next[:,:,:,2] = torch.minimum(torch.abs(w_next[:,:,:,2]), lam[2]) * torch.exp(1j*torch.angle(w_next[:,:,:,2]))
                        
                    v_est = w_next + j/(j+3) * (w_next - w_est)
                    w_est = w_next
                    
                x_next = u - gam*DT(w_est)
                
        # Nesterov extrapolation
        z_est = x_next + (i/(i+3)) * (x_next - x_est)
        
        x_est = x_next
        
        # display if needed
        if display and i % display_iter == 0:
            if len(x_est.shape) == 2:
                x_disp = x_est.detach().cpu()
            else:
                x_disp = x_est[:,:,0].detach().cpu()
            
            ax1 = plt.subplot(1,2,1)
            plt.imshow(torch.abs(x_disp),cmap='gray',interpolation='nearest')
            plt.axis('off')
            ax2 = plt.subplot(1,2,2)
            plt.imshow(torch.angle(x_disp),cmap='gray',interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
    runtime = time.time()-time_start
    if verbose:
        print('Time spent: {:.2f} s'.format(runtime))
    
    # move results back to cpu
    x_est = x_est.cpu()
    y     = y.cpu()
    masks = masks.cpu()
    HQ1   = HQ1.cpu()
    HQ2   = HQ2.cpu()
    
    x_est_tv = x_est
    
    # display results and calculate errors
    x_ref = x
    
    metrics_amp_tv = 0
    metrics_pha_tv = 0
    
    for k in range(K):
        
        amp_est = torch.abs(  C2(C1((x_est[:,:,math.ceil(k*K_recon/K)]))))
        pha_est = torch.angle(C2(C1((x_est[:,:,math.ceil(k*K_recon/K)]))))
        amp_ref = torch.abs(  C2(C1((x_ref[:,:,math.ceil(k*K_recon/K)]))))
        pha_ref = torch.angle(C2(C1((x_ref[:,:,math.ceil(k*K_recon/K)]))))
        
        # normalize phase to avoid global ambiguity
        pha_est = pha_est - torch.mean(pha_est) + torch.mean(pha_ref)
        x_est_crp = amp_est * torch.exp(1j*pha_est)
        x_ref_crp = amp_ref * torch.exp(1j*pha_ref)    
        
        # calculate amplitude and phase psnr values as error metrics
        metrics_amp_tv += psnr(amp_ref.numpy(), amp_est.numpy())
        metrics_pha_tv += psnr((pha_ref.numpy()+PI)/2/PI, (pha_est.numpy()+PI)/2/PI)
        
        plt.subplot(2,2,1)
        plt.imshow(amp_est,vmin=0.5, vmax=1.0,cmap='gray',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(pha_est,vmin=0,vmax=PI/2,cmap='inferno',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(amp_est-amp_ref,vmin=-0.2, vmax=0.2,cmap='seismic',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(pha_est-pha_ref,vmin=-0.5,vmax=0.5,cmap='seismic',interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    metrics_amp_tv /= K
    metrics_pha_tv /= K
    
    # save results as a .mat file
    if save_results:
        o_filename = 'cache_sim/results_'+clipname+'_'+str(fr_start)+'_'+'{:.3f}'.format(noise_level)+'_'+denoiser_seq+'.mat'
        scipy.io.savemat(o_filename,{'x_est': x_est.numpy()})
    
    
    #====================================================
    # reconstruction algorithm (denoiser: DRUNet + TV)
    #====================================================
    
    K_recon = K     # number of reconstructed time points
    KK = lambda x: np.floor(K_recon*k/K).astype(int)    # round to integer indices
    
    x_est = x_est_tv    # initial estimate
    
    # load the wrapping function for DRUNet
    myImageDenoiserDRUNet = ImageDenoiserDRUNet(model_path="./models/drunet/drunet.pth", color_mode='gray', device=device)
    
    # define the denoising operators in sequence
    denoiser_seq = 'drunet+tv'
    denoisers = denoiser_seq.split('+')
    
    # define the regularization parameters
    lams = []
    for denoiser in denoisers:
        # TV denoiser
        if denoiser.lower() == 'tv':
            
            def lam(x):
                lam_start = np.array([1e-3 + 0.1*noise_level, 1e-3 + 0.1*noise_level, 1e-3 + 0.1*noise_level])
                lam_end   = np.array([1e-3, 1e-3, 1e-3])
                return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
            lams.append(lam)
        
        # DRUNet denoiser
        elif denoiser.lower() == 'drunet':
            
            def lam(x):
                lam_start = 1e-4 + 2e-2*noise_level
                lam_end   = 0.5e-4 + 1e-2*noise_level
                return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
            lams.append(lam)
            
    # algorithm parameters
    gam = 2         # step size
    n_iters = 20    # number of iterations
    n_subiters = 5  # number of subiterations for the TV-based proximal update
    
    # define auxiliary variables
    z_est = x_est
    g_est = torch.zeros(x_est.shape).type(torch.complex64)
    v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
    w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
    
    x_est     = x_est.to(device)
    y         = y.to(device)
    masks     = masks.to(device)
    HQ1       = HQ1.to(device)
    HQ2       = HQ2.to(device)
    g_est     = g_est.to(device)
    z_est     = z_est.to(device)
    v_est     = v_est.to(device)
    w_est     = w_est.to(device)
    
    time_start = time.time()
    
    verbose = False
    display = False
    display_iter = 1
    
    # main loop
    for i in range(n_iters):
        
        if verbose:
            print('Iter {:0>4d} / {:0>4d}'.format(i+1, n_iters))
        
        # gradient update
        g_est[:] = 0
        for k in range(K):
            u = A(z_est[:,:,KK(k)],k)
            u = (torch.abs(u) - torch.sqrt(y[:,:,k])) * torch.exp(1j*torch.angle(u))
            g_est[:,:,KK(k)] = g_est[:,:,KK(k)] + 1/2/(K/K_recon) * AH(u,k)
        
        u = z_est - gam * g_est
        
        # proximal update
        for idx, denoiser in enumerate(denoisers):
            lam = lams[idx](i)
            if denoiser == 'drunet':
                sig = 20*lam*gam
                u = denoise_drunet(u, myImageDenoiserDRUNet, sig, scale=SCALE)
            elif denoiser == 'tv':
                lam = torch.Tensor(lam).to(device)
                v_est[:] = 0
                w_est[:] = 0
                for j in range(n_subiters):
                    w_next = v_est + 1/12/gam * D(u-gam*DT(v_est))
                    w_next[:,:,:,0] = torch.minimum(torch.abs(w_next[:,:,:,0]), lam[0]) * torch.exp(1j*torch.angle(w_next[:,:,:,0]))
                    w_next[:,:,:,1] = torch.minimum(torch.abs(w_next[:,:,:,1]), lam[1]) * torch.exp(1j*torch.angle(w_next[:,:,:,1]))
                    w_next[:,:,:,2] = torch.minimum(torch.abs(w_next[:,:,:,2]), lam[2]) * torch.exp(1j*torch.angle(w_next[:,:,:,2]))
                        
                    v_est = w_next + j/(j+3) * (w_next - w_est)
                    w_est = w_next
                    
                u = u - gam*DT(w_est)
        
        x_next = u
        
        # Nesterov extrapolation
        z_est = x_next + (i/(i+3)) * (x_next - x_est)
        
        x_est = x_next
        
        # display if needed
        if display and i % display_iter == 0:
            if len(x_est.shape) == 2:
                x_disp = x_est.detach().cpu()
            else:
                x_disp = x_est[:,:,0].detach().cpu()
            
            ax1 = plt.subplot(1,2,1)
            plt.imshow(torch.abs(x_disp),cmap='gray',interpolation='nearest')
            plt.axis('off')
            ax2 = plt.subplot(1,2,2)
            plt.imshow(torch.angle(x_disp),cmap='gray',interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        
    runtime = time.time()-time_start
    if verbose:
        print('Time spent: {:.2f} s'.format(runtime))
    
    # move results back to cpu
    x_est = x_est.cpu()
    y     = y.cpu()
    masks = masks.cpu()
    HQ1   = HQ1.cpu()
    HQ2   = HQ2.cpu()
    
    
    # display results and calculate errors
    x_ref = x
    
    metrics_amp_drunet = 0
    metrics_pha_drunet = 0
    
    for k in range(K):
        
        amp_est = torch.abs(  C2(C1((x_est[:,:,math.ceil(k*K_recon/K)]))))
        pha_est = torch.angle(C2(C1((x_est[:,:,math.ceil(k*K_recon/K)]))))
        amp_ref = torch.abs(  C2(C1((x_ref[:,:,math.ceil(k*K_recon/K)]))))
        pha_ref = torch.angle(C2(C1((x_ref[:,:,math.ceil(k*K_recon/K)]))))
        
        # normalize phase to avoid global ambiguity
        pha_est = pha_est - torch.mean(pha_est) + torch.mean(pha_ref)
        x_est_crp = amp_est * torch.exp(1j*pha_est)
        x_ref_crp = amp_ref * torch.exp(1j*pha_ref)
        
        # calculate amplitude and phase psnr values as error metrics
        metrics_amp_drunet += psnr(amp_ref.numpy(), amp_est.numpy())
        metrics_pha_drunet += psnr((pha_ref.numpy()+PI)/2/PI, (pha_est.numpy()+PI)/2/PI)
        
        plt.subplot(2,2,1)
        plt.imshow(amp_est,vmin=0.5, vmax=1.0,cmap='gray',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(pha_est,vmin=0,vmax=PI/2,cmap='inferno',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(amp_est-amp_ref,vmin=-0.2, vmax=0.2,cmap='seismic',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(pha_est-pha_ref,vmin=-0.5,vmax=0.5,cmap='seismic',interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    metrics_amp_drunet /= K
    metrics_pha_drunet /= K
    
    # save results as a .mat file
    if save_results:
        o_filename = 'cache_sim/results_'+clipname+'_'+str(fr_start)+'_'+'{:.3f}'.format(noise_level)+'_'+denoiser_seq+'.mat'
        scipy.io.savemat(o_filename,{'x_est': x_est.numpy()})
    
    
    #========================================================
    # reconstruction algorithm (denoiser: FastDVDnet + TV)
    #========================================================
    
    K_recon = K     # number of reconstructed time points
    KK = lambda x: np.floor(K_recon*k/K).astype(int)    # round to integer indices

    x_est = x_est_tv    # initial estimate
    
    # load the wrapping function for FastDVDnet
    myVideoDenoiserFastDVDnet = VideoDenoiserFastDVDnet(model_path="./models/fastdvdnet/fastdvdnet.pth", device=device)
    
    # define the denoising operators in sequence
    denoiser_seq = 'fastdvdnet+tv'
    denoisers = denoiser_seq.split('+')
    
    # define the regularization parameters
    lams = []
    for denoiser in denoisers:
        # TV denoiser
        if denoiser.lower() == 'tv':
            
            def lam(x):
                lam_start = np.array([1e-3 + 0.1*noise_level, 1e-3 + 0.1*noise_level, 1e-3 + 0.1*noise_level])
                lam_end   = np.array([1e-3, 1e-3, 1e-3])
                return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
            lams.append(lam)
        
        # FastDVDNet denoiser
        elif denoiser.lower() == 'fastdvdnet':
            
            def lam(x):
                lam_start = 1e-4 + 2e-2*noise_level
                lam_end   = 0.5e-4 + 1e-2*noise_level
                return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
            lams.append(lam)
            
    # algorithm parameters
    gam = 2         # step size
    n_iters = 20    # number of iterations
    n_subiters = 5  # number of subiterations for the TV-based proximal update
    
    # define auxiliary variables
    z_est = x_est
    g_est = torch.zeros(x_est.shape).type(torch.complex64)
    v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
    w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
    
    x_est     = x_est.to(device)
    y         = y.to(device)
    masks     = masks.to(device)
    HQ1       = HQ1.to(device)
    HQ2       = HQ2.to(device)
    g_est     = g_est.to(device)
    z_est     = z_est.to(device)
    v_est     = v_est.to(device)
    w_est     = w_est.to(device)
    
    time_start = time.time()
    
    verbose = False
    display = False
    display_iter = 1
    
    # main loop
    for i in range(n_iters):
        
        if verbose:
            print('Iter {:0>4d} / {:0>4d}'.format(i+1, n_iters))
        
        # gradient update
        g_est[:] = 0
        for k in range(K):
            u = A(z_est[:,:,KK(k)],k)
            u = (torch.abs(u) - torch.sqrt(y[:,:,k])) * torch.exp(1j*torch.angle(u))
            g_est[:,:,KK(k)] = g_est[:,:,KK(k)] + 1/2/(K/K_recon) * AH(u,k)
        
        u = z_est - gam * g_est
        
        # proximal update
        for idx, denoiser in enumerate(denoisers):
            lam = lams[idx](i)
            if denoiser == 'fastdvdnet':
                sig = 20*lam*gam
                u = denoise_fastdvdnet(u, myVideoDenoiserFastDVDnet, sig, scale=SCALE)
            elif denoiser == 'tv':
                lam = torch.Tensor(lam).to(device)
                v_est[:] = 0
                w_est[:] = 0
                for j in range(n_subiters):
                    w_next = v_est + 1/12/gam * D(u-gam*DT(v_est))
                    w_next[:,:,:,0] = torch.minimum(torch.abs(w_next[:,:,:,0]), lam[0]) * torch.exp(1j*torch.angle(w_next[:,:,:,0]))
                    w_next[:,:,:,1] = torch.minimum(torch.abs(w_next[:,:,:,1]), lam[1]) * torch.exp(1j*torch.angle(w_next[:,:,:,1]))
                    w_next[:,:,:,2] = torch.minimum(torch.abs(w_next[:,:,:,2]), lam[2]) * torch.exp(1j*torch.angle(w_next[:,:,:,2]))
                        
                    v_est = w_next + j/(j+3) * (w_next - w_est)
                    w_est = w_next
                    
                u = u - gam*DT(w_est)
        
        x_next = u
        
        # Nesterov extrapolation
        z_est = x_next + (i/(i+3)) * (x_next - x_est)
        
        x_est = x_next
        
        # display if needed
        if display and i % display_iter == 0:
            if len(x_est.shape) == 2:
                x_disp = x_est.detach().cpu()
            else:
                x_disp = x_est[:,:,0].detach().cpu()
            
            ax1 = plt.subplot(1,2,1)
            plt.imshow(torch.abs(x_disp),cmap='gray',interpolation='nearest')
            plt.axis('off')
            ax2 = plt.subplot(1,2,2)
            plt.imshow(torch.angle(x_disp),cmap='gray',interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        
    runtime = time.time()-time_start
    if verbose:
        print('Time spent: {:.2f} s'.format(runtime))
    
    # move results back to cpu
    x_est = x_est.cpu()
    y     = y.cpu()
    masks = masks.cpu()
    HQ1   = HQ1.cpu()
    HQ2   = HQ2.cpu()
    
    
    # display results and calculate errors
    x_ref = x
    
    metrics_amp_dvdnet = 0
    metrics_pha_dvdnet = 0
    
    for k in range(K):
        
        amp_est = torch.abs(  C2(C1((x_est[:,:,math.ceil(k*K_recon/K)]))))
        pha_est = torch.angle(C2(C1((x_est[:,:,math.ceil(k*K_recon/K)]))))
        amp_ref = torch.abs(  C2(C1((x_ref[:,:,math.ceil(k*K_recon/K)]))))
        pha_ref = torch.angle(C2(C1((x_ref[:,:,math.ceil(k*K_recon/K)]))))
        
        # normalize phase to avoid global ambiguity
        pha_est = pha_est - torch.mean(pha_est) + torch.mean(pha_ref)
        x_est_crp = amp_est * torch.exp(1j*pha_est)
        x_ref_crp = amp_ref * torch.exp(1j*pha_ref)
        
        # calculate amplitude and phase psnr values as error metrics
        metrics_amp_dvdnet += psnr(amp_ref.numpy(), amp_est.numpy())
        metrics_pha_dvdnet += psnr((pha_ref.numpy()+PI)/2/PI, (pha_est.numpy()+PI)/2/PI)
        
        plt.subplot(2,2,1)
        plt.imshow(amp_est,vmin=0.5, vmax=1.0,cmap='gray',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(pha_est,vmin=0,vmax=PI/2,cmap='inferno',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(amp_est-amp_ref,vmin=-0.2, vmax=0.2,cmap='seismic',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(pha_est-pha_ref,vmin=-0.5,vmax=0.5,cmap='seismic',interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    metrics_amp_dvdnet /= K
    metrics_pha_dvdnet /= K
    
    # save results as a .mat file
    if save_results:
        o_filename = 'cache_sim/results_'+clipname+'_'+str(fr_start)+'_'+'{:.3f}'.format(noise_level)+'_'+denoiser_seq+'.mat'
        scipy.io.savemat(o_filename,{'x_est': x_est.numpy()})
    
    
    #=====================================================
    # reconstruction algorithm (denoiser: ViDNet + TV)
    #=====================================================
    
    K_recon = K     # number of reconstructed time points
    KK = lambda x: np.floor(K_recon*k/K).astype(int)    # round to integer indices

    x_est = x_est_tv    # initial estimate
    
    # load the wrapping function for ViDNet
    myVideoDenoiserViDNet = VideoDenoiserViDNet(model_path="./models/vidnet/vidnet.pth", device=device)
    
    # load the wrapping function for FastDVDnet
    denoiser_seq = 'vidnet+tv'
    denoisers = denoiser_seq.split('+')
    
    # define the regularization parameters
    lams = []
    for denoiser in denoisers:
        # TV denoiser
        if denoiser.lower() == 'tv':
            
            def lam(x):
                lam_start = np.array([1e-3 + 0.1*noise_level, 1e-3 + 0.1*noise_level, 1e-3 + 0.1*noise_level])
                lam_end   = np.array([1e-3, 1e-3, 1e-3])
                return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
            lams.append(lam)
            
        # VidNet denoiser
        elif denoiser.lower() == 'vidnet':
            
            def lam(x):
                lam_start = 1e-4 + 2e-2*noise_level
                lam_end   = 0.5e-4 + 1e-2*noise_level
                return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
            lams.append(lam)
            
    # algorithm parameters
    gam = 2         # step size
    n_iters = 20    # number of iterations
    n_subiters = 5  # number of subiterations for the TV-based proximal update
    
    # define auxiliary variables
    z_est = x_est
    g_est = torch.zeros(x_est.shape).type(torch.complex64)
    v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
    w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
    
    x_est     = x_est.to(device)
    y         = y.to(device)
    masks     = masks.to(device)
    HQ1       = HQ1.to(device)
    HQ2       = HQ2.to(device)
    g_est     = g_est.to(device)
    z_est     = z_est.to(device)
    v_est     = v_est.to(device)
    w_est     = w_est.to(device)
    
    time_start = time.time()
    
    verbose = False
    display = False
    display_iter = 1
    
    # main loop
    for i in range(n_iters):
        
        if verbose:
            print('Iter {:0>4d} / {:0>4d}'.format(i+1, n_iters))
        
        # gradient update
        g_est[:] = 0
        for k in range(K):
            u = A(z_est[:,:,KK(k)],k)
            u = (torch.abs(u) - torch.sqrt(y[:,:,k])) * torch.exp(1j*torch.angle(u))
            g_est[:,:,KK(k)] = g_est[:,:,KK(k)] + 1/2/(K/K_recon) * AH(u,k)
        
        u = z_est - gam * g_est
        
        # proximal update
        for idx, denoiser in enumerate(denoisers):
            lam = lams[idx](i)
            if denoiser == 'vidnet':
                sig = 20*lam*gam
                u = denoise_vidnet(u, myVideoDenoiserViDNet, sig, scale=SCALE)
            elif denoiser == 'tv':
                lam = torch.Tensor(lam).to(device)
                v_est[:] = 0
                w_est[:] = 0
                for j in range(n_subiters):
                    w_next = v_est + 1/12/gam * D(u-gam*DT(v_est))
                    w_next[:,:,:,0] = torch.minimum(torch.abs(w_next[:,:,:,0]), lam[0]) * torch.exp(1j*torch.angle(w_next[:,:,:,0]))
                    w_next[:,:,:,1] = torch.minimum(torch.abs(w_next[:,:,:,1]), lam[1]) * torch.exp(1j*torch.angle(w_next[:,:,:,1]))
                    w_next[:,:,:,2] = torch.minimum(torch.abs(w_next[:,:,:,2]), lam[2]) * torch.exp(1j*torch.angle(w_next[:,:,:,2]))
                        
                    v_est = w_next + j/(j+3) * (w_next - w_est)
                    w_est = w_next
                    
                u = u - gam*DT(w_est)
        
        x_next = u
        
        # Nesterov extrapolation
        z_est = x_next + (i/(i+3)) * (x_next - x_est)
        
        x_est = x_next
        
        # display if needed
        if display and i % display_iter == 0:
            if len(x_est.shape) == 2:
                x_disp = x_est.detach().cpu()
            else:
                x_disp = x_est[:,:,0].detach().cpu()
            
            ax1 = plt.subplot(1,2,1)
            plt.imshow(torch.abs(x_disp),cmap='gray',interpolation='nearest')
            plt.axis('off')
            ax2 = plt.subplot(1,2,2)
            plt.imshow(torch.angle(x_disp),cmap='gray',interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        
    runtime = time.time()-time_start
    if verbose:
        print('Time spent: {:.2f} s'.format(runtime))
    
    # move results back to cpu
    x_est = x_est.cpu()
    y     = y.cpu()
    masks = masks.cpu()
    HQ1   = HQ1.cpu()
    HQ2   = HQ2.cpu()
    
    
    # display results and calculate errors
    x_ref = x
    
    metrics_amp_vidnet = 0
    metrics_pha_vidnet = 0
    
    for k in range(K):
        
        amp_est = torch.abs(  C2(C1((x_est[:,:,math.ceil(k*K_recon/K)]))))
        pha_est = torch.angle(C2(C1((x_est[:,:,math.ceil(k*K_recon/K)]))))
        amp_ref = torch.abs(  C2(C1((x_ref[:,:,math.ceil(k*K_recon/K)]))))
        pha_ref = torch.angle(C2(C1((x_ref[:,:,math.ceil(k*K_recon/K)]))))
        
        # normalize phase to avoid global ambiguity
        pha_est = pha_est - torch.mean(pha_est) + torch.mean(pha_ref)
        x_est_crp = amp_est * torch.exp(1j*pha_est)
        x_ref_crp = amp_ref * torch.exp(1j*pha_ref)
        
        # calculate amplitude and phase psnr values as error metrics
        metrics_amp_vidnet += psnr(amp_ref.numpy(), amp_est.numpy())
        metrics_pha_vidnet += psnr((pha_ref.numpy()+PI)/2/PI, (pha_est.numpy()+PI)/2/PI)
        
        plt.subplot(2,2,1)
        plt.imshow(amp_est,vmin=0.5, vmax=1.0,cmap='gray',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(pha_est,vmin=0,vmax=PI/2,cmap='inferno',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(amp_est-amp_ref,vmin=-0.2, vmax=0.2,cmap='seismic',interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(pha_est-pha_ref,vmin=-0.5,vmax=0.5,cmap='seismic',interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    metrics_amp_vidnet /= K
    metrics_pha_vidnet /= K
    
    # save results as a .mat file
    if save_results:
        o_filename = 'cache_sim/results_'+clipname+'_'+str(fr_start)+'_'+'{:.3f}'.format(noise_level)+'_'+denoiser_seq+'.mat'
        scipy.io.savemat(o_filename,{'x_est': x_est.numpy()})
    
    
    #=============================
    # print results
    #=============================
    
    data = [[clipname, 'TV', metrics_amp_tv, metrics_pha_tv],
            [clipname, 'TV + DRUNet', metrics_amp_drunet, metrics_pha_drunet],
            [clipname, 'TV + FastDVDnet', metrics_amp_dvdnet, metrics_pha_dvdnet],
            [clipname, 'TV + ViDNet', metrics_amp_vidnet, metrics_pha_vidnet]]
    headers = ['Clipname', 'Denoiser', 'Amplitude PSNR (dB)', 'Phase PSNR (dB)']
    print(tabulate(data, headers=headers, tablefmt="grid"))
    