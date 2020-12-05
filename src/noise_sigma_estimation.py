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
    
    :return (float): the estimated noise sigma of a given picture
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

    :images_path(string):
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

    return est_sigma_sum/n_success


def denoise_images(images_path, avg_sigma=0.1, save_dir='./denoised_imgs/'):
    os.makedirs(save_dir, exist_ok=True)

    print('Denoising:')
    n_frames = len(glob.glob(images_path + '*' + IMG_EXTENTION))
    with tqdm(total=n_frames) as pbar:
        for n_processed, img_path in enumerate(glob.glob(images_path + '*' + IMG_EXTENTION)):
            img = imread(img_path)

            res_img = denoise_wavelet(img, multichannel=True, convert2ycbcr=True,
                                        method='VisuShrink', mode='soft',
                                        sigma=avg_sigma, rescale_sigma=True)
            res_img_name = save_dir + img_path.split('/')[-1]
            res_img = img_as_ubyte(res_img)
            imsave(res_img_name, res_img)
            if (n_processed % 10 == 0):
                pbar.update(10)


if __name__ == '__main__':
    # video_path = '/run/media/alex/Data/video_enhancment/Demo2/hockey_1_7GB.mov'
    # video_path = '/home/alex/Desktop/hockey17_DVD_sig30_DR_cut.mp4'
    # video_path = '/run/media/alex/Data/video_enhancment/IVHC/videoplayback.mp4'
    # cut_frames(video_path, frames_num=10)
    # avg_sigma, avr_variance, avr_degree, avr_level = avg_images_stats("./noisy_video_frames/")

    est_sigma = avg_images_noise_sigma('/run/media/alex/Data/video_enh_framework/video-processing/joined_frames/')
    print(f'est_sigma = {est_sigma}')


    # print()
    # print(f'avr_sigma = {avg_sigma}')
    # print(f'avr_variance = {avr_variance}')
    # print(f'avr_degree = {avr_degree}')
    # print(f'avr_level = {avr_level}')

    # test_denoise_methods('./noisy_video_frames/001.png')
    # denoise_images("./noisy_video_frames/", avg_sigma=avg_sigma, save_dir='./denoised_imgs/')
