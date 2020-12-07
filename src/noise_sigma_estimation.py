import os
import glob
from ivhc.imnest_ivhc import imnest_ivhc
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.restoration import estimate_sigma, denoise_wavelet
import numpy as np
import ffmpeg
from tqdm import tqdm

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio


IMG_EXTENTION = '.png'  


def get_est_sigma(image_path, method='skimage', max_poly_deg=5):
    '''
    Image noise estimation

    :image_path (string): path to the image
    :method (string, optional): can be either ivhc or skimage
    :poly (int, optional): max polynomial regression degree for ivhc 
    
    :return (float): estimated noise sigma of a given picture 
                        or None if ivhc throws exception due to low sigma
    '''
    img = imread(image_path)
    print(f'{image_path}')

    est_sigma = None

    if method == 'skimage':
        est_sigma = estimate_sigma(img, multichannel=True, average_sigmas=True)
    elif method == 'ivhc':
        try:
           est_sigma, noise_degree, noise_level_function = imnest_ivhc(img, max_poly_deg)
        except Exception as e:
            print(f'Low sigma caused division by zero. Try other method of estimation.')

    return est_sigma


def avg_images_noise_sigma(images_path, method='skimage', max_poly_deg=5):
    '''
    Multiple image noise estimation

    :images_path(string): path to images
    :method (string, optional): can be either ivhc or skimage
    :poly (int, optional): max polynomial regression degree for ivhc method

    :return (float): average estimated noise sigma
    '''
    est_sigma_sum, n_success = 0, 0

    images_path += '*' + IMG_EXTENTION

    print('Estimating:')
    for img_path in glob.glob(images_path):
        est_sigma = get_est_sigma(img_path, method, max_poly_deg)
        if est_sigma is not None:
            est_sigma_sum += est_sigma
            n_success += 1

    if n_success <= 0:
        return None

    return est_sigma_sum / n_success
