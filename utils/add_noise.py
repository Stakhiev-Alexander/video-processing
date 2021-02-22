from glob import glob

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

IMG_EXTENTION = '.png'


def add_noise_img(sigma, img_path):
    image = imread(img_path)

    gaussian = np.random.normal(0, sigma, image.shape)
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image


def add_noise_imgs(imgs_path='./video_frames/', out_path='./noisy_video_frames/', sigma=30):
    imgs_path += '*' + IMG_EXTENTION

    for n_processed, img_path in enumerate(tqdm(glob.glob(imgs_path))):
        n_img = add_noise_img(sigma, img_path)
        imsave(out_path + img_path.split('/')[-1], n_img)


if __name__ == '__main__':
    add_noise_imgs()
