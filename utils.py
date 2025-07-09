import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as mpl_param
from PIL import Image
import glob
import os
import cv2
IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types
PI = math.pi
mpl_param["figure.dpi"] = 500

def propagate(input_wave, dist, pxsize, wavlen):

    # calculate parameters
    k = 2*PI/wavlen
    n1 = input_wave.shape[0]
    n2 = input_wave.shape[1]
    
    # define meshgrid
    k1 = PI/pxsize*torch.linspace(-1, 1-2/n1, n1)
    k2 = PI/pxsize*torch.linspace(-1, 1-2/n2, n2)
    kk1, kk2 = torch.meshgrid(k1, k2, indexing='ij')
    
    # circular convolution via FFTs
    output_wave = torch.fft.ifft2(torch.fft.ifftshift(
                  torch.exp(1j*dist*torch.sqrt(k**2 - kk1**2 - kk2**2))
                  * torch.fft.fftshift(torch.fft.fft2(input_wave))))
    
    return output_wave

    
def cropimg(x, cropsize):
    if cropsize > 0:
        y = x[cropsize:-cropsize,cropsize:-cropsize]
    else:
        y = x
    return y


def padimg(x, padsize, padval=0):
    y = F.pad(x, (padsize,padsize,padsize,padsize), mode='constant', value=padval)
    return y


def imshift(x, shift):
    n1, n2 = x.shape[0], x.shape[1]
    f1 = torch.linspace(-n1/2,n1/2-1,n1)
    f2 = torch.linspace(-n2/2,n2/2-1,n2)
    u1, u2 = torch.meshgrid(f1, f2, indexing='ij')
    y  = torch.fft.ifft2(torch.fft.fftshift(
        torch.exp(-1j*2*PI*(shift[0]*u1/n1 + shift[1]*u2/n2))
        * torch.fft.fftshift(torch.fft.fft2(x))))
    return y


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def load_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.LANCZOS)
            
    return torch.tensor(np.asarray(img)/255.)


def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
    r""" Opens a sequence of images and expands it to even sizes if necesary
    Args:
        fpath: string, path to image sequence
        gray_mode: boolean, True indicating if images is to be open are in grayscale mode
        expand_if_needed: if True, the spatial dimensions will be expanded if
            size is odd
        expand_axis0: if True, output will have a fourth dimension
        max_num_fr: maximum number of frames to load
    Returns:
        seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
            The image gets normalized gets normalized to the range [0, 1].
        expanded_h: True if original dim H was odd and image got expanded in this dimension.
        expanded_w: True if original dim W was odd and image got expanded in this dimension.
    """
    # Get ordered list of filenames
    files = get_imagenames(seq_dir)

    seq_list = []
    # print("\tOpen sequence in folder: ", seq_dir)
    for fpath in files[0:max_num_fr]:

        img, expanded_h, expanded_w = open_image(fpath,\
                                                   gray_mode=gray_mode,\
                                                   expand_if_needed=expand_if_needed,\
                                                   expand_axis0=False)
            
        seq_list.append(img)
    seq = np.stack(seq_list, axis=0)
    return seq, expanded_h, expanded_w

    
def get_imagenames(seq_dir, pattern=None):
    """ Get ordered list of filenames
    """
    files = []
    for typ in IMAGETYPES:
        files.extend(glob.glob(os.path.join(seq_dir, typ)))

    # filter filenames
    if not pattern is None:
        ffiltered = []
        ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
        files = ffiltered
        del ffiltered

    # sort filenames alphabetically
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files


def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
    r""" Opens an image and expands it if necesary
    Args:
        fpath: string, path of image file
        gray_mode: boolean, True indicating if image is to be open
            in grayscale mode
        expand_if_needed: if True, the spatial dimensions will be expanded if
            size is odd
        expand_axis0: if True, output will have a fourth dimension
    Returns:
        img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
            if expand_axis0=False, the output will have a shape CxHxW.
            The image gets normalized gets normalized to the range [0, 1].
        expanded_h: True if original dim H was odd and image got expanded in this dimension.
        expanded_w: True if original dim W was odd and image got expanded in this dimension.
    """
    if not gray_mode:
        # Open image as a CxHxW torch.Tensor
        img = cv2.imread(fpath)
        # from HxWxC to CxHxW, RGB image
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

    if expand_axis0:
        img = np.expand_dims(img, 0)

    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = img.shape
    if expand_if_needed:
        if sh_im[-2]%2 == 1:
            expanded_h = True
            if expand_axis0:
                img = np.concatenate((img, \
                    img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
            else:
                img = np.concatenate((img, \
                    img[:, -1, :][:, np.newaxis, :]), axis=1)


        if sh_im[-1]%2 == 1:
            expanded_w = True
            if expand_axis0:
                img = np.concatenate((img, \
                    img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
            else:
                img = np.concatenate((img, \
                    img[:, :, -1][:, :, np.newaxis]), axis=2)

    if normalize_data:
        img = normalize(img)
    return img, expanded_h, expanded_w


def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data/255.)


def visualize_complex(cimg, cmap, amp_range, mode, reverse):

    amp = np.abs(cimg)
    pha = np.angle(cimg)
    
    amin = amp_range[0]
    amax = amp_range[1]
    
    amp_norm = (amp-amin)/(amax-amin)
    
    b = 1
    
    ncmap = cmap.shape[0]
    img = np.zeros((cimg.shape[0], cimg.shape[1] ,3))
    
    for i in range (cimg.shape[0]):
        for j in range(cimg.shape[1]):
            a = amp_norm[i,j]
            if reverse:
                a = 1-a

            if mode.lower() == 'hsv':
                img[i,j,:] = cmap[1+round((ncmap-1)/2/PI*(pha[i,j]+PI)),:] * a
            elif mode.lower() == 'hsl':
                if a > 1/2:
                    w = (2*(a-1/2))**b
                    img[i,j,:] = cmap[round((ncmap-1)/2/PI*(pha[i,j]+PI)),:] * (1-w) + np.array([1,1,1]) * w
                else:
                    w = (2*(1/2-a))**b
                    img[i,j,:] = cmap[round((ncmap-1)/2/PI*(pha[i,j]+PI)),:] * (1-w) + np.array([0,0,0]) * w

    
    n = 256
    cbarimg = np.zeros((n,n,3))
    x = np.linspace(-1,1,n)
    y = x
    X, Y = np.meshgrid(x,y,indexing='xy')
    theta, rho = cart2pol(X,Y)
    for i in range(n):
        for j in range(n):
            if rho[i,j] <= 1:
                a = rho[i,j]
                if reverse:
                    a = 1-a
                if mode.lower() == 'hsv':
                    cbarimg[i,j,:] = cmap[round((ncmap-1)/2/PI*(theta[i,j]+PI)),:] * a
                elif mode.lower() == 'hsl':
                    if a > 1/2:
                        w = (2*(a-1/2))**b
                        cbarimg[i,j,:] = cmap[round((ncmap-1)/2/PI*(theta[i,j]+PI)),:] * (1-w) + np.array([1,1,1]) * w
                    else:
                        w = (2*(1/2-a))**b
                        cbarimg[i,j,:] = cmap[round((ncmap-1)/2/PI*(theta[i,j]+PI)),:] * (1-w) + np.array([0,0,0]) * w


    return img, cbarimg


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def sinebow(n):
    h = np.linspace(0,1,n)
    h = h + 1/2;
    h = h * (-1)
    r = np.sin(PI*h)
    g = np.sin(PI*(h+1/3))
    b = np.sin(PI*(h+2/3));
    c = np.concatenate((r.reshape(n,1),g.reshape(n,1),b.reshape(n,1)),axis=1)
    c = c**2
    return c