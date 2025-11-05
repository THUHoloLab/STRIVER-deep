# =============================================================================
# This code provides a demonstration of video plug-and-play (PnP) algorithm for
# dynamic holographic reconstruction based on experimental data.
# 
# Author:   Yunhui Gao
# Date:     2025/06/10
# =============================================================================


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as mpl_param
from utils import *
from phantominator import shepp_logan
from torchvision import transforms
from PIL import Image
from models.drunet.wrapper import *
from models.fastdvdnet.wrapper import *
from models.fastdvdnet.utils import *
import models.drunet.utils_image as util
import models.drunet.utils_model
from models.vidnet.wrapper import *
import scipy
import time
import cmocean
import os


mpl_param["figure.dpi"] = 500


# =============================================================================
# define constants and auxiliary functions
# =============================================================================

# define constants
PI = math.pi        # math constant
DEVICE_INDEX = 0    # gpu index
SCALE = 1.6         # scaling factor


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
    n1 = in_shape[0]
    n2 = in_shape[1]
    
    # define meshgrid
    k1 = PI/pxsize*torch.linspace(-1, 1-2/n1, n1)
    k2 = PI/pxsize*torch.linspace(-1, 1-2/n2, n2)
    kk1, kk2 = torch.meshgrid(k1, k2, indexing='ij')
    
    # calculate transfer function
    H = torch.exp(1j*dist*torch.sqrt(k**2 - kk1**2 - kk2**2))
    
    return H


def D(x):
    '''
    Calculate the gradient (finite difference) field of a 2D / 3D datacube.
    
    Parameters
    ----------
    x : torch.Tensor
        2D image / 3D video.

    Returns
    -------
    g : torch.Tensor
        4D gradient tensor.
    '''
    # if x is a 3D array (video sequence)
    if x.shape[2] > 1:
        g1 = x - torch.cat((x[1:,:,:], x[-1,None,:,:]), 0)
        g2 = x - torch.cat((x[:,1:,:], x[:,-1,None,:]), 1)
        g3 = x - torch.cat((x[:,:,1:], x[:,:,-1,None]), 2)
    
    # if x is a 2D array (image)
    else:
        g1 = x - torch.cat((x[1:,:,:], x[-1,None,:,:]), 0)
        g2 = x - torch.cat((x[:,1:,:], x[:,-1,None,:]), 1)
        g3 = x - x
    
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
    
    # temporal axis
    if g.shape[2] > 1:
        u3 = g[:,:,:,2] - torch.roll(g[:,:,:,2],1,2)
        u3[:,:,0] = g[:,:,0,2]
        u3[:,:,-1] = -g[:,:,-2,2]
    else:
        u3 = 0
    
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


# =============================================================================
# main function
# =============================================================================

if __name__ == "__main__":
    
    device = torch.device("cuda:{}".format(DEVICE_INDEX) if torch.cuda.is_available() else "cpu")
    
    # select dataset
    exp_num = 10
    grp_num = 4
    
    # define data batch size
    frame_step = 5
    frame_ref = 1
    
    # define data batch size
    K = 10      # number of measurements within a batch
    
    for frame_start in [1]:
        for speed in range(0,10):
            
            torch.cuda.empty_cache()
            
            print('frame start: {:0>4d} | speed : {:0>4d}'.format(frame_start, speed))
             
            
            # =================================================================
            # load data
            # =================================================================
            
            print('Load data ...')
            
            # load diffuser calibration data
            filename = './data/exp/E'+str(exp_num)+'/G'+str(grp_num)+'/calib/calib_diffuser.mat'
            data = scipy.io.loadmat(filename)
            diffuser = torch.tensor(data['diffuser'])
            bias_1, bias_2 = data['bias_1'][0,0], data['bias_2'][0,0]
            bias_1 = np.uint16(bias_1)
            bias_2 = np.uint16(bias_2)
            sizeout_1, sizeout_2 = data['sizeout_1'][0,0], data['sizeout_2'][0,0]
            prefix = data['prefix'][0]
            params = data['params'][0,0]
            pxsize = params['pxsize'][0,0]
            wavlen = params['wavlen'][0,0]
            dist_1 = params['dist_1'][0,0]
            dist_2 = params['dist_2'][0,0]
            nullpixels_2 = int(data['cropsize_2'][0,0])
            
            # load translation position calibration data
            filename = './data/exp/E'+str(exp_num)+'/G'+str(grp_num)+'/calib/calib_shift.mat'
            data = scipy.io.loadmat(filename)
            shifts = data['shifts']
            shifts = np.flipud(shifts)
            
            # pre-specified area of interest
            rect_aoi_image = np.round([1196.51,727.51,1169.98,1559.98]).astype(int)

            nullpixels_1 = 50
            
            # spatial dimension of the image
            Mh = rect_aoi_image[3]
            Mw = rect_aoi_image[2]
            
            # crop rectangle applied to the diffuser
            rect_aoi_diffuser = [rect_aoi_image[0], rect_aoi_image[1],\
                                 rect_aoi_image[2]+2*nullpixels_2, rect_aoi_image[3]+2*nullpixels_2]  
            
            # extract relative translation positions (total shift range)
            shift_ref = shifts[:,frame_ref-1]
            shifts = shifts[:,frame_start-1:frame_start+K*frame_step-1:frame_step] - shift_ref.reshape(2,1)
            
            # introduce virtual sample translation
            speed_1 = speed
            speed_2 = 0
                        
            shifts[0,:] = shifts[0,:] + np.linspace(-speed_1*K/2,speed_1*(K/2-1),K)
            shifts[1,:] = shifts[1,:] + np.linspace(-speed_2*K/2,speed_2*(K/2-1),K)
            
            # calculate the model parameters for each measurement
            y, diffusers = [], []
            for k in range(K):
                
                # load the raw intensity image
                filename = './data/exp/E'+str(exp_num)+'/G'+str(grp_num)+'/'+prefix+str(frame_start+k*frame_step)+'.bmp'
                img_obj = padimg_2(load_image(filename),[bias_1,bias_2],[sizeout_1,sizeout_2])
                img_obj  = torch.abs(imshift(img_obj, (shifts[0,k], shifts[1,k])))
                y_tmp = cropimg_2(img_obj,rect_aoi_image)
                y.append(torch.unsqueeze(y_tmp, dim=2))
                
                # calculate the translated diffuser profile
                diff = imshift(diffuser, (shifts[0,k], shifts[1,k]));
                diff = cropimg_2(torch.abs(diff),rect_aoi_diffuser) * torch.exp(1j*cropimg_2(torch.angle(diff),rect_aoi_diffuser))
                diffusers.append(torch.unsqueeze(diff, dim=2))
            
            y = torch.cat(y, dim=2)
            diffusers = torch.cat(diffusers, dim=2)
            
            
            # =================================================================
            # define the forward model
            # =================================================================
            
            Nh = Mh + 2*nullpixels_2 + 2*nullpixels_1
            Nw = Mw + 2*nullpixels_2 + 2*nullpixels_1
            
            # calcualte the transfer functions for free-space propagation
            HQ1 = torch.fft.fftshift(transfunc_propagate((Nh, Nw), dist_1, pxsize, wavlen))
            HQ2 = torch.fft.fftshift(transfunc_propagate((Mh+2*nullpixels_2, Mw+2*nullpixels_2), dist_2, pxsize, wavlen))
            
            Q1  = lambda x: torch.fft.ifft2(torch.fft.fft2(x) * HQ1)                # free-space propagation from sample to diffuser
            Q1H = lambda x: torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(HQ1))    # Hermitian operator of Q1
            C1  = lambda x: cropimg(x, nullpixels_1)                                # image cropping
            C1T = lambda x: padimg(x, nullpixels_1)                                 # transpose (Hermitian) operator of C1
            M   = lambda x, k: x * diffusers[:,:,k]                                 # diffuser modulation
            MH  = lambda x, k: x * torch.conj(diffusers[:,:,k])                     # Hermitian operator of M
            Q2  = lambda x: torch.fft.ifft2(torch.fft.fft2(x) * HQ2)                # free-space propagation from diffuser to sensor
            Q2H = lambda x: torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(HQ2))    # Hermitian of Q2
            C2  = lambda x: cropimg(x, nullpixels_2)                                # image cropping
            C2T = lambda x: padimg(x, nullpixels_2)                                 # transpose (Hermitian) operator of C2
            A   = lambda x, k: C2(Q2(M(C1(Q1(x)),k)))                               # overall measurement operator
            AH  = lambda x, k: Q1H(C1T(MH(Q2H(C2T(x)),k)))                          # Hermitian of A
            
            K_recon = K     # number of reconstructed time points
            KK = lambda x: np.floor(K_recon*k/K).astype(int)    # round to integer indices
            
            # output folder
            o_foldername = 'results/e'+str(exp_num)+'g'+str(grp_num)
            if not os.path.exists(o_foldername):
                os.makedirs(o_foldername)
            
            
            # =================================================================
            # 3DTV (uniform initialization)
            # =================================================================
            
            print('=========================================')
            print('      3DTV (uniform initialization)      ')
            print('=========================================')
            
            # optimized regularization parameters
            if speed == 0:
                reg = 1e-2
            elif speed == 1:
                reg = 3e-3
            elif speed == 2:
                reg = 1e-3
            else:
                reg = 3e-4
            
            # initial estimate
            x_est = torch.ones(Nh,Nw,K_recon).type(torch.complex64)
            
            # define the denoising operators in sequence
            denoiser_seq = 'tv3d'
            denoisers = denoiser_seq.split('+')
            
            # define the regularization parameters
            lams = []
            for denoiser in denoisers:
                
                # TV denoiser
                if denoiser.lower() == 'tv3d':
                    
                    def lam(x):
                        lam_start = np.array([1e-1, 1e-1, 1e-1])    # three-element array for x, y, and t regularization
                        lam_end   = np.array([1e-3, 1e-3, reg])
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                
            # algorithm parameters
            gam = 2. / torch.max(torch.abs(diffusers))**2   # step size
            n_iters = 200       # number of iterations
            n_subiters = 10     # number of subiterations for the TV-based proximal update
            
            # define auxiliary variables
            z_est = x_est
            g_est = torch.zeros(x_est.shape).type(torch.complex64)
            v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            
            x_est     = x_est.to(device)
            y         = y.to(device)
            diffusers = diffusers.to(device)
            HQ1       = HQ1.to(device)
            HQ2       = HQ2.to(device)
            g_est     = g_est.to(device)
            z_est     = z_est.to(device)
            v_est     = v_est.to(device)
            w_est     = w_est.to(device)
            
            time_start = time.time()
            
            # main loop
            for i in range(n_iters):
                
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
                    if denoiser == 'tv3d':
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
                
                runtime = time.time()-time_start
                print('iter: {:0>4d} / {:0>4d}  |  runtime: {:8.2f} s'.format(i+1, n_iters, runtime))
            
            # move results back to cpu
            x_est     = x_est.cpu()
            y         = y.cpu()
            diffusers = diffusers.cpu()
            HQ1       = HQ1.cpu()
            HQ2       = HQ2.cpu()
            
            # display results
            for k in range(K):
                amp_est = torch.abs(  C2(C1(x_est[:,:,math.floor(k*K_recon/K)])))
                pha_est = torch.angle(C2(C1(x_est[:,:,math.floor(k*K_recon/K)])))
                ax1 = plt.subplot(1,2,1)
                plt.imshow(amp_est,cmap='gray',interpolation='nearest')
                plt.axis('off')
                ax2 = plt.subplot(1,2,2)
                plt.imshow(pha_est,vmin=-PI,vmax=PI,cmap='inferno',interpolation='nearest')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # save results as a .mat file
            o_filename = 'results_'+denoiser_seq+'_'+str(frame_start)+'_'+str(speed_1)+'.mat'
            scipy.io.savemat(o_foldername+'/'+o_filename,{'x_est': x_est.numpy()})
            
            
            # =================================================================
            # DRUNet + 3DTV (3DTV initialization)
            # =================================================================
            
            print('=========================================')
            print('   DRUNet + 3DTV (3DTV initialization)   ')
            print('=========================================')
            
            # stage 1: 3DTV reconstruction
            
            # optimized regularization parameters
            if speed == 0:
                reg = 1e-2
            elif speed == 1:
                reg = 3e-3
            elif speed <= 7:
                reg = 1e-3
            else:
                reg = 3e-4
                
            # initial estimate
            x_est = torch.ones(Nh,Nw,K_recon).type(torch.complex64)
            
            # define the denoising operators in sequence
            denoiser_seq = 'tv3d'
            denoisers = denoiser_seq.split('+')
            
            # define the regularization parameters
            lams = []
            for denoiser in denoisers:
                
                # TV denoiser
                if denoiser.lower() == 'tv3d':
                    
                    def lam(x):
                        lam_start = np.array([1e-1, 1e-1, 1e-1])    # three-element array for x, y, and t regularization
                        lam_end   = np.array([1e-3, 1e-3, reg])
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
            
            # algorithm parameters
            gam = 2. / torch.max(torch.abs(diffusers))**2   # step size
            n_iters = 200       # number of iterations
            n_subiters = 10     # number of subiterations for the TV-based proximal update
            
            # define auxiliary variables
            z_est = x_est
            g_est = torch.zeros(x_est.shape).type(torch.complex64)
            v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            
            x_est     = x_est.to(device)
            y         = y.to(device)
            diffusers = diffusers.to(device)
            HQ1       = HQ1.to(device)
            HQ2       = HQ2.to(device)
            g_est     = g_est.to(device)
            z_est     = z_est.to(device)
            v_est     = v_est.to(device)
            w_est     = w_est.to(device)
            
            time_start = time.time()
            
            # main loop
            for i in range(n_iters):
                
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
                    if denoiser == 'tv3d':
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
                
                runtime = time.time()-time_start
                print('iter: {:0>4d} / {:0>4d}  |  runtime: {:8.2f} s'.format(i+1, n_iters, runtime))
                
            # move results back to cpu
            x_est     = x_est.cpu()
            y         = y.cpu()
            diffusers = diffusers.cpu()
            HQ1       = HQ1.cpu()
            HQ2       = HQ2.cpu()
            
            
            # stage 2: DRUNet + 3DTV reconstruction
            
            # load the wrapping function for DRUNet
            myImageDenoiserDRUNet = ImageDenoiserDRUNet(model_path="./models/drunet/drunet.pth", color_mode='gray', device=device)
            
            # define the denoising operators in sequence
            denoiser_seq = 'drunet+tv3d'
            denoisers = denoiser_seq.split('+')
            
            # define the regularization parameters
            lams = []
            for denoiser in denoisers:
                
                # TV denoiser
                if denoiser.lower() == 'tv3d':
                    
                    def lam(x):
                        lam_start = np.array([1e-3, 1e-3, reg])     # three-element array for x, y, and t regularization
                        lam_end   = np.array([3e-4, 3e-4, reg])
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                
                # DRUNet denoiser
                elif denoiser.lower() == 'drunet':
                    
                    def lam(x):
                        lam_start = 2.5e-3
                        lam_end   = 2.5e-4
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                    
            # algorithm parameters
            gam = 2. / torch.max(torch.abs(diffusers))**2   # step size
            n_iters = 20        # number of iterations
            n_subiters = 10     # number of subiterations for the TV-based proximal update
            
            # define auxiliary variables
            z_est = x_est
            g_est = torch.zeros(x_est.shape).type(torch.complex64)
            v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            
            x_est     = x_est.to(device)
            y         = y.to(device)
            diffusers = diffusers.to(device)
            HQ1       = HQ1.to(device)
            HQ2       = HQ2.to(device)
            g_est     = g_est.to(device)
            z_est     = z_est.to(device)
            v_est     = v_est.to(device)
            w_est     = w_est.to(device)
            
            time_start = time.time()
            
            # main loop
            for i in range(n_iters):
                
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
                        sig = 20*lam*torch.sqrt(gam)
                        u = denoise_drunet(u, myImageDenoiserDRUNet, sig, scale=SCALE)
                    elif denoiser == 'tv3d':
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
                
                runtime = time.time()-time_start
                print('iter: {:0>4d} / {:0>4d}  |  runtime: {:8.2f} s'.format(i+1, n_iters, runtime))
                    
            # move results back to cpu
            x_est     = x_est.cpu()
            y         = y.cpu()
            diffusers = diffusers.cpu()
            HQ1       = HQ1.cpu()
            HQ2       = HQ2.cpu()
            
            # display results
            for k in range(K):
                amp_est = torch.abs(  C2(C1(x_est[:,:,math.floor(k*K_recon/K)])))
                pha_est = torch.angle(C2(C1(x_est[:,:,math.floor(k*K_recon/K)])))
                ax1 = plt.subplot(1,2,1)
                plt.imshow(amp_est,cmap='gray',interpolation='nearest')
                plt.axis('off')
                ax2 = plt.subplot(1,2,2)
                plt.imshow(pha_est,vmin=-PI,vmax=PI,cmap='inferno',interpolation='nearest')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # save results as a .mat file
            o_filename = 'results_'+denoiser_seq+'_'+str(frame_start)+'_'+str(speed_1)+'.mat'
            scipy.io.savemat(o_foldername+'/'+o_filename,{'x_est': x_est.numpy()})
            
            
            # =================================================================
            # FastDVDnet + 3DTV (3DTV initialization)
            # =================================================================
            
            print('=========================================')
            print(' FastDVDnet + 3DTV (3DTV initialization) ')
            print('=========================================')
            
            # stage 1: 3DTV reconstruction
            
            # initial estimate
            x_est = torch.ones(Nh,Nw,K_recon).type(torch.complex64)
            
            # define the denoising operators in sequence
            denoiser_seq = 'tv3d'
            denoisers = denoiser_seq.split('+')
            
            # define the regularization parameters
            lams = []
            for denoiser in denoisers:
                
                # TV denoiser
                if denoiser.lower() == 'tv3d':
                    
                    def lam(x):
                        lam_start = np.array([1e-1, 1e-1, 1e-1])    # three-element array for x, y, and t regularization
                        lam_end   = np.array([1e-3, 1e-3, 1e-3])
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                
            # algorithm parameters 
            gam = 2. / torch.max(torch.abs(diffusers))**2   # step size
            n_iters = 200       # number of iterations
            n_subiters = 10     # number of subiterations for the TV-based proximal update
            
            # define auxiliary variables
            z_est = x_est
            g_est = torch.zeros(x_est.shape).type(torch.complex64)
            v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            
            x_est     = x_est.to(device)
            y         = y.to(device)
            diffusers = diffusers.to(device)
            HQ1       = HQ1.to(device)
            HQ2       = HQ2.to(device)
            g_est     = g_est.to(device)
            z_est     = z_est.to(device)
            v_est     = v_est.to(device)
            w_est     = w_est.to(device)
            
            time_start = time.time()
            
            # main loop
            for i in range(n_iters):
                
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
                    if denoiser == 'tv3d':
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
                
                runtime = time.time()-time_start
                print('iter: {:0>4d} / {:0>4d}  |  runtime: {:8.2f} s'.format(i+1, n_iters, runtime))
                
            # move results back to cpu
            x_est     = x_est.cpu()
            y         = y.cpu()
            diffusers = diffusers.cpu()
            HQ1       = HQ1.cpu()
            HQ2       = HQ2.cpu()
            
            
            # stage 2: FastDVDnet + 3DTV reconstruction
            
            # load the wrapping function for FastDVDnet
            myVideoDenoiserFastDVDnet = VideoDenoiserFastDVDnet(model_path="./models/fastdvdnet/fastdvdnet.pth", device=device)
            
            # define the denoising operators in sequence
            denoiser_seq = 'fastdvdnet+tv3d'
            denoisers = denoiser_seq.split('+')
            
            # define the regularization parameters
            lams = []
            for denoiser in denoisers:
                
                # TV denoiser
                if denoiser.lower() == 'tv3d':
                    
                    def lam(x):
                        lam_start = np.array([1e-3, 1e-3, 1e-3])    # three-element array for x, y, and t regularization
                        lam_end   = np.array([3e-4, 3e-4, 3e-4])
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                    
                # FastDVDnet denoiser
                elif denoiser.lower() == 'fastdvdnet':
                    
                    def lam(x):
                        lam_start = 2.5e-3
                        lam_end   = 2.5e-4
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                    
            # algorithm parameters
            gam = 2. / torch.max(torch.abs(diffusers))**2   # step size
            n_iters = 30        # number of iterations
            n_subiters = 10     # number of subiterations for the TV-based proximal update
            
            # define auxiliary variables
            z_est = x_est
            g_est = torch.zeros(x_est.shape).type(torch.complex64)
            v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            
            x_est     = x_est.to(device)
            y         = y.to(device)
            diffusers = diffusers.to(device)
            HQ1       = HQ1.to(device)
            HQ2       = HQ2.to(device)
            g_est     = g_est.to(device)
            z_est     = z_est.to(device)
            v_est     = v_est.to(device)
            w_est     = w_est.to(device)
            
            time_start = time.time()
            
            # main loop
            for i in range(n_iters):
                
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
                        sig = 20*lam*torch.sqrt(gam)
                        u = denoise_fastdvdnet(u, myVideoDenoiserFastDVDnet, sig, scale=SCALE)
                    elif denoiser == 'tv3d':
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
                
                runtime = time.time()-time_start
                print('iter: {:0>4d} / {:0>4d}  |  runtime: {:8.2f} s'.format(i+1, n_iters, runtime))
                
            # move results back to cpu
            x_est     = x_est.cpu()
            y         = y.cpu()
            diffusers = diffusers.cpu()
            HQ1       = HQ1.cpu()
            HQ2       = HQ2.cpu()
            
            # display results
            for k in range(K):
                amp_est = torch.abs(  C2(C1(x_est[:,:,math.floor(k*K_recon/K)])))
                pha_est = torch.angle(C2(C1(x_est[:,:,math.floor(k*K_recon/K)])))
                ax1 = plt.subplot(1,2,1)
                plt.imshow(amp_est,cmap='gray',interpolation='nearest')
                plt.axis('off')
                ax2 = plt.subplot(1,2,2)
                plt.imshow(pha_est,vmin=-PI,vmax=PI,cmap='inferno',interpolation='nearest')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
            # save results as a .mat file
            o_filename = 'results_'+denoiser_seq+'_'+str(frame_start)+'_'+str(speed_1)+'.mat'
            scipy.io.savemat(o_foldername+'/'+o_filename,{'x_est': x_est.numpy()})
            
            
            # =================================================================
            # ViDNet + 3DTV (3DTV initialization)
            # =================================================================
            
            print('=========================================')
            print('   ViDNet + 3DTV (3DTV initialization)   ')
            print('=========================================')
            
            # stage 1: 3DTV reconstruction
            
            # initial estimate
            x_est = torch.ones(Nh,Nw,K_recon).type(torch.complex64)
            
            # define the denoising operators in sequence
            denoiser_seq = 'tv3d'
            denoisers = denoiser_seq.split('+')
            
            # define the regularization parameters
            lams = []
            for denoiser in denoisers:
                
                # TV denoiser
                if denoiser.lower() == 'tv3d':
                    
                    def lam(x):
                        lam_start = np.array([1e-1, 1e-1, 1e-1])    # three-element array for x, y, and t regularization
                        lam_end   = np.array([1e-3, 1e-3, 1e-3])
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                
            # algorithm parameters 
            gam = 2. / torch.max(torch.abs(diffusers))**2   # step size
            n_iters = 200       # number of iterations
            n_subiters = 10     # number of subiterations for the TV-based proximal update
            
            # define auxiliary variables
            z_est = x_est
            g_est = torch.zeros(x_est.shape).type(torch.complex64)
            v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            
            x_est     = x_est.to(device)
            y         = y.to(device)
            diffusers = diffusers.to(device)
            HQ1       = HQ1.to(device)
            HQ2       = HQ2.to(device)
            g_est     = g_est.to(device)
            z_est     = z_est.to(device)
            v_est     = v_est.to(device)
            w_est     = w_est.to(device)
            
            time_start = time.time()
            
            # main loop
            for i in range(n_iters):
                
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
                    if denoiser == 'tv3d':
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
                
                runtime = time.time()-time_start
                print('iter: {:0>4d} / {:0>4d}  |  runtime: {:8.2f} s'.format(i+1, n_iters, runtime))
                
            # move results back to cpu
            x_est     = x_est.cpu()
            y         = y.cpu()
            diffusers = diffusers.cpu()
            HQ1       = HQ1.cpu()
            HQ2       = HQ2.cpu()
            
            
            # stage 2: ViDNet + 3DTV reconstruction
            
            # load the wrapping function for ViDNet
            myVideoDenoiserViDNet = VideoDenoiserViDNet(model_path="./models/vidnet/vidnet.pth", device=device)
            
            # define the denoising operators in sequence
            denoiser_seq = 'vidnet+tv3d'
            denoisers = denoiser_seq.split('+')
            
            # define the regularization parameters
            lams = []
            for denoiser in denoisers:
                
                # TV denoiser
                if denoiser.lower() == 'tv3d':
                    
                    def lam(x):
                        lam_start = np.array([1e-3, 1e-3, 1e-3])    # three-element array for x, y, and t regularization
                        lam_end   = np.array([3e-4, 3e-4, 3e-4])
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                    
                # ViDNet denoiser
                elif denoiser.lower() == 'vidnet':
                    
                    def lam(x):
                        lam_start = 2.5e-3
                        lam_end   = 2.5e-4
                        return reg_param(lam_start, lam_end, x, n_iters, alpha=5)
                    lams.append(lam)
                    
            # algorithm parameters 
            gam = 2. / torch.max(torch.abs(diffusers))**2   # step size
            n_iters = 30        # number of iterations
            n_subiters = 10     # number of subiterations for the TV-based proximal update
            
            # define auxiliary variables
            z_est = x_est
            g_est = torch.zeros(x_est.shape).type(torch.complex64)
            v_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            w_est = torch.zeros((x_est.shape[0], x_est.shape[1], x_est.shape[2], 3)).type(torch.complex64)
            
            x_est     = x_est.to(device)
            y         = y.to(device)
            diffusers = diffusers.to(device)
            HQ1       = HQ1.to(device)
            HQ2       = HQ2.to(device)
            g_est     = g_est.to(device)
            z_est     = z_est.to(device)
            v_est     = v_est.to(device)
            w_est     = w_est.to(device)
            
            time_start = time.time()
            
            # main loop
            for i in range(n_iters):
                
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
                        sig = 20*lam*torch.sqrt(gam)
                        u = denoise_vidnet(u, myVideoDenoiserViDNet, sig, scale=SCALE)
                    elif denoiser == 'tv3d':
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
                
                runtime = time.time()-time_start
                print('iter: {:0>4d} / {:0>4d}  |  runtime: {:8.2f} s'.format(i+1, n_iters, runtime))
                
            # move results back to cpu
            x_est     = x_est.cpu()
            y         = y.cpu()
            diffusers = diffusers.cpu()
            HQ1       = HQ1.cpu()
            HQ2       = HQ2.cpu()
            
            # display results
            for k in range(K):
                amp_est = torch.abs(  C2(C1(x_est[:,:,math.floor(k*K_recon/K)])))
                pha_est = torch.angle(C2(C1(x_est[:,:,math.floor(k*K_recon/K)])))
                ax1 = plt.subplot(1,2,1)
                plt.imshow(amp_est,cmap='gray',interpolation='nearest')
                plt.axis('off')
                ax2 = plt.subplot(1,2,2)
                plt.imshow(pha_est,vmin=-PI,vmax=PI,cmap='inferno',interpolation='nearest')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
            # save results as a .mat file
            o_filename = 'results_'+denoiser_seq+'_'+str(frame_start)+'_'+str(speed_1)+'.mat'
            scipy.io.savemat(o_foldername+'/'+o_filename,{'x_est': x_est.numpy()})
            