import argparse
import os
import pathlib
import shutil
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

sys.path.append(sys.path[0] + '/..')

from utils.scene_detection import find_scenes
import utils.logger as logger
from nets.DeepLab.infer_prob import infer_dl
from nets.RIFE.inference_imgs import infer_rife, get_rife_model

from nets.flownet.infer_flownet import infer_flownet

logger = logger.get_logger(__name__)
base_path = str(pathlib.Path(__file__).parent.absolute())


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
    mask = ((ang_diff > 0.9) & (ang_diff < 1.1)) & ((mag1 > 3) | (mag2 > 3))
    return np.where(mask, 255, 0)  # True if moving object, else False


def combine_rife_dl_flownet(mask_path, flow_path, orig_img_path, inter_img_path, out_path):
    logger.info("Starting combining")
    ext = "/*.png"
    i = 1

    for flow, mask, orig, inter in tzip(sorted(glob(flow_path + ext)), sorted(glob(mask_path + ext))[1:],
                                        sorted(glob(orig_img_path + ext))[2:],
                                        sorted(glob(inter_img_path + ext))[2::]):
        i += 1
        flow = cv2.imread(flow, 0)
        mask = cv2.imread(mask, 0) & np.logical_not(flow)
        orig = cv2.imread(orig, 0)
        inter = cv2.imread(inter, 0)
        out = np.where(mask, inter, orig)
        cv2.imwrite(out_path + '/' + str(i).zfill(6) + ".png", out)


def combine_rife_dl(mask_path, orig_img_path, inter_img_path, out_path):
    logger.info("Starting combining")
    ext = "/*.png"
    i = 1

    for mask, orig, inter in tzip(sorted(glob(mask_path + ext))[1:], sorted(glob(orig_img_path + ext))[2:],
                                  sorted(glob(inter_img_path + ext))[2::]):
        i += 1
        mask = cv2.imread(mask, 0)
        orig = cv2.imread(orig, 0)
        inter = cv2.imread(inter, 0)
        out = np.where(mask, inter, orig)
        cv2.imwrite(out_path + '/' + str(i).zfill(6) + ".png", out)


def rife_stage(args):
    first_inter = base_path + "/../output/first_inter/"
    shutil.rmtree(first_inter, ignore_errors=True)
    shutil.rmtree(args.rife_out, ignore_errors=True)
    os.makedirs(first_inter, exist_ok=True)
    os.makedirs(args.rife_out, exist_ok=True)

    slices = find_scenes(args.in_path, return_slices=True)
    imgs_paths = sorted(glob(args.in_path + '/*.png'))

    logger.info("Starting double interpolation")
    model = get_rife_model()
    with tqdm(total=len(imgs_paths)) as pbar:
        for s in slices:
            if isinstance(s, tuple):
                in_paths = imgs_paths[slice(*s)]

                infer_rife(in_paths=in_paths, out_path=first_inter, keep_source_imgs=False, model=model, tqdm_bar=pbar)
                in_paths2 = sorted(glob(first_inter + '/*.png'))
                infer_rife(in_paths=in_paths2, out_path=args.rife_out, keep_source_imgs=False, starting_index=s[0] + 1,
                           model=model)

                shutil.rmtree(first_inter, ignore_errors=True)
                os.makedirs(first_inter, exist_ok=True)

                shutil.copy(imgs_paths[s[0]], args.rife_out + str(s[0]).zfill(6) + ".png")
                shutil.copy(imgs_paths[s[1] - 1], args.rife_out + str(s[1] - 1).zfill(6) + ".png")
            else:  # 1 frame
                shutil.copy(imgs_paths[s], args.rife_out)
                pbar.update(1)


def dl_stage(args):
    shutil.rmtree(args.dl_out, ignore_errors=True)
    os.makedirs(args.dl_out, exist_ok=True)
    logger.info("Starting deeplab")
    infer_dl(args.in_path, args.dl_out)


def flownet_stage(args):
    flownet_forward = base_path + "/../output/flownet_forward/"
    flownet_reverse = base_path + "/../output/flownet_reverse/"
    shutil.rmtree(flownet_forward, ignore_errors=True)
    shutil.rmtree(flownet_reverse, ignore_errors=True)
    shutil.rmtree(args.flownet_out, ignore_errors=True)
    os.makedirs(flownet_forward, exist_ok=True)
    os.makedirs(flownet_reverse, exist_ok=True)
    os.makedirs(args.flownet_out, exist_ok=True)

    logger.info("Starting forward flownet")
    infer_flownet(args.dl_out, flownet_forward, reverse=False, downscale_factor=args.downscale_factor)
    logger.info("Starting reverse flownet")
    infer_flownet(args.dl_out, flownet_reverse, reverse=True, downscale_factor=args.downscale_factor)

    front = sorted(glob(flownet_forward + "*.flo"))[1:]
    back = sorted(glob(flownet_reverse + "*.flo"))[::-1]

    for i, (f1, f2) in enumerate(tzip(front, back)):
        combined_flows = two_way_flow(f1, f2)

        if args.downscale_factor != 1:
            h, w = combined_flows.shape[0] * args.downscale_factor, combined_flows.shape[1] * args.downscale_factor
            combined_flows = combined_flows.astype('float32')
            combined_flows = cv2.resize(combined_flows, dsize=(w, h), interpolation=cv2.INTER_LINEAR).astype('int32')

        cv2.imwrite(args.flownet_out + str(i).zfill(6) + ".png", combined_flows)

    shutil.rmtree(flownet_forward, ignore_errors=True)
    shutil.rmtree(flownet_reverse, ignore_errors=True)


if __name__ == '__main__':
    logger.info("Starting artefacts stage")

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', '-i', type=str, help='image to test')
    parser.add_argument('--out-path', '-o', type=str, help='mask image to save')
    parser.add_argument('--downscale-factor', '-f', type=int, default=1)
    parser.add_argument('--rife-out', type=str, default="/../output/rife_out/")
    parser.add_argument('--dl-out', type=str, default="/../output/dl_out/")
    parser.add_argument('--flownet-out', type=str, default="/../output/flownet_out/")
    parser.add_argument('--use-dl', action='store_true')
    parser.add_argument('--use-dl-flownet', action='store_true')

    args = parser.parse_args()

    args.rife_out = base_path + args.rife_out
    shutil.rmtree(args.out_path, ignore_errors=True)
    os.makedirs(args.out_path, exist_ok=True)

    rife_stage(args)

    if args.use_dl_flownet:
        args.dl_out = base_path + args.dl_out
        args.flownet_out = base_path + args.flownet_out

        dl_stage(args)
        flownet_stage(args)

        combine_rife_dl_flownet(args.dl_out, args.flownet_out, args.in_path, args.rife_out, args.out_path)
    elif args.use_dl:
        args.dl_out = base_path + args.dl_out
        combine_rife_dl(args.dl_out, args.in_path, args.rife_out, args.out_path)
    else:
        shutil.rmtree(args.out_path, ignore_errors=True)
        os.rename(args.rife_out, args.out_path)

    # shutil.rmtree(args.rife_out, ignore_errors=True)
    # shutil.rmtree(args.dl_out, ignore_errors=True)
    # shutil.rmtree(args.flownet_out, ignore_errors=True)

    logger.info("Finished artefacts stage")
