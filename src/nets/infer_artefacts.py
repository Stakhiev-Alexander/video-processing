import argparse
import os
import shutil
from glob import glob

import cv2
import numpy as np
from ..utils import logger as ps_logger
from DeepLab.infer_prob import infer_dl
from flownet.infer_flownet import infer_flownet
from tqdm.contrib import tzip

from src.nets.RIFE.inference_imgs import infer_rife

logger = ps_logger.get_logger(__name__)


def read_flow(fn):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            return np.resize(data, (int(h), int(w), 2))


def two_way_flow(f1, f2):
    flow1 = read_flow(f1)
    flow2 = read_flow(f2)

    mag1, ang1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
    mag2, ang2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
    ang_diff = abs(ang1 - ang2) / np.pi
    mask = ((ang_diff > 0.8) & (ang_diff < 1.2)) & ((mag1 > 3) | (mag2 > 3))
    return np.where(mask, 255, 0)  # True if moving object, else False


def combine_masks_with_2xint(mask_path, flow_path, orig_img_path, inter_img_path, out_path):
    ext = "/*.png"
    i = 1

    for flow, mask, orig, inter in tzip(glob(flow_path + ext), glob(mask_path + ext)[1:], glob(orig_img_path + ext)[2:],
                                        glob(inter_img_path + ext)[2::2]):
        i += 1
        flow = cv2.imread(flow, 0)
        mask = (cv2.imread(mask, 0) & np.logical_not(flow))
        orig = cv2.imread(orig, 0)
        inter = cv2.imread(inter, 0)
        out = np.where(mask, inter, orig)
        cv2.imwrite(out_path + str(i).zfill(6) + ".png", out)


if __name__ == '__main__':
    logger.info("Starting artefacts stage")
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', '-i', type=str, help='image to test')
    parser.add_argument('--out-path', '-o', type=str, help='mask image to save')
    args = parser.parse_args()

    first_inter = "first_inter/"
    second_inter = "second_inter/"

    shutil.rmtree(first_inter, ignore_errors=True)
    shutil.rmtree(second_inter, ignore_errors=True)
    shutil.rmtree(args.out_path, ignore_errors=True)

    os.makedirs(first_inter, exist_ok=True)
    os.makedirs(second_inter, exist_ok=True)
    os.makedirs(args.out_path, exist_ok=True)

    logger.info("Starting first interpolation")
    infer_rife(in_path=args.in_path, out_path=first_inter, keep_source_imgs=False, starting_index=1)
    logger.info("Starting second interpolation")
    infer_rife(in_path=first_inter, out_path=second_inter, keep_source_imgs=True, starting_index=1)

    shutil.copy(sorted(os.listdir(args.in_path))[0], second_inter + str(0).zfill(6) + ".png")
    shutil.copy(sorted(os.listdir(args.in_path))[-1],
                second_inter + str(len(os.listdir(second_inter))).zfill(6) + ".png")

    shutil.rmtree(first_inter, ignore_errors=True)

    dl_out = "dl_out/"
    shutil.rmtree(dl_out, ignore_errors=True)
    os.makedirs(dl_out, exist_ok=True)
    logger.info("Starting deeplab")
    infer_dl(args.in_path, dl_out)

    flownet_forward = "flownet_forward/"
    flownet_reverse = "flownet_reverse/"
    flownet_out = "flownet_out/"
    shutil.rmtree(flownet_forward, ignore_errors=True)
    shutil.rmtree(flownet_reverse, ignore_errors=True)
    shutil.rmtree(flownet_out, ignore_errors=True)
    os.makedirs(flownet_forward, exist_ok=True)
    os.makedirs(flownet_reverse, exist_ok=True)
    os.makedirs(flownet_out, exist_ok=True)

    logger.info("Starting forward flownet")
    infer_flownet(dl_out, flownet_forward, reverse=False)
    logger.info("Starting reverse flownet")
    infer_flownet(dl_out, flownet_reverse, reverse=True)

    front = glob(flownet_forward + "*.flo")[1:]
    back = glob(flownet_reverse + "*.flo")[::-1]

    for i, (f1, f2) in enumerate(tzip(front, back)):
        cv2.imwrite(flownet_out + str(i).zfill(6) + ".png", two_way_flow(f1, f2))

    shutil.rmtree(flownet_forward, ignore_errors=True)
    shutil.rmtree(flownet_reverse, ignore_errors=True)

    logger.info("Starting combining")
    combine_masks_with_2xint(dl_out, flownet_out, args.in_path, second_inter, args.out_path)

    shutil.rmtree(flownet_out, ignore_errors=True)
    shutil.rmtree(second_inter, ignore_errors=True)
    shutil.rmtree(dl_out, ignore_errors=True)

    logger.info("Finished artefacts stage")
