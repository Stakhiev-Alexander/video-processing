import os

from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim


def _plot_figures(figures, nrows, ncols):
    """Plot a dictionary of figures.

    :figures (dict<str, numpy.ndarray>): <title, figure> dictionary
    :ncols (int): number of columns of subplots wanted in the display
    :nrows (int): number of rows of subplots wanted in the figure
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.show()


def get_ssim(original_img_path, compare_img_paths):
    """
    :original_img_path (str): the path to the picture with which we will compare
    :compare_img_paths (list of str): a list of paths to images that will be compared with the original

    :return (dict{str: float}): mapped image's paths and SSIMs
    """
    original_img = img_as_float(imread(original_img_path))

    compare_imgs_ssim = {}

    for img_path in compare_img_paths:
        comp_img = img_as_float(imread(img_path))
        compare_imgs_ssim[img_path] = ssim(original_img, comp_img, multichannel=True,
                  data_range=comp_img.max() - comp_img.min())


    return compare_imgs_ssim


def plot_ssim_imgs(original_img_path, compare_img_paths, nrows, ncols):
    """
    :original_img_path (str): the path to the picture with which we will compare
    :compare_img_paths (list of str): a list of paths to images that will be compared with the original
    :ncols (int): number of columns of subplots wanted in the display
    :nrows (int): number of rows of subplots wanted in the figure
    """

    figures = {}
    original_img = img_as_float(imread(original_img_path))
    figures['Original image'] = original_img
    print(type(original_img))

    compare_imgs_ssim = get_ssim(original_img_path, compare_img_paths)

    for img_path, ssim in compare_imgs_ssim.items():
        img_name = os.path.basename(img_path)
        title = f'{img_name}; SSIM = {ssim:.2f}'
        figures[title] = img_as_float(imread(img_path))

    _plot_figures(figures, nrows, ncols)    


if __name__ == '__main__':
    original_img_path = './compare_imgs/orig.png'
    compare_img_paths = ['./compare_imgs/noisy_s10.png',
                            './n10_FastDVDnet_0.png',
                            './n15_FastDVDnet_0.png']

    plot_ssim_imgs(original_img_path, compare_img_paths, nrows=2, ncols=2)
