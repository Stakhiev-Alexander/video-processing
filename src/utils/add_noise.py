import glob
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm


IMG_EXTENTION = '.png'


def add_noise_img(sigma, img_path, clipped=False):
    image =  imread(img_path)

    gaussian = np.random.normal(0,sigma,image.shape)
    noisy_image = image + gaussian
    if clipped:
        noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image


def add_noise_imgs(imgs_path='./video_frames/', out_path='./noisy_video_frames/', sigma=30):
    imgs_path += '*' + IMG_EXTENTION
    np.random.seed(0)
    with tqdm(total=len(glob.glob(imgs_path))) as pbar:
        for n_processed, img_path in enumerate(glob.glob(imgs_path)):
            n_img = add_noise_img(sigma, img_path)
            imsave(out_path + img_path.split('/')[-1], n_img)
            pbar.update(1)


if __name__ == '__main__':
    add_noise_imgs()
