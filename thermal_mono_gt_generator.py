import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import tqdm
import sys
from config import *

from model.sgbm import StereoBlockMatching


def load_single_file_with_pattern(module, frame_num, lens_id):
    f = glob.glob(f'{module.replace("l0", f"l{lens_id}")}_{frame_num}_*')
    if len(f) != 1:
        # print(f"Expected 1 file for {module}_{frame_num}. Found {len(f)}.")
        return None
    return f[0]

def preprocess_nir(nir_path, gt_res=0.5):
    if nir_path is None:
        return nir_path
    img = cv2.imread(nir_path, cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    img = cv2.resize(img, None,fx=gt_res, fy=gt_res, interpolation=cv2.INTER_LINEAR)
    img = np.rot90(img)
    return (img * (255 / img.max())).astype(np.uint8)

def preprocess_thermal(thermal_path, gt_res=0.5):
    if thermal_path is None:
        return None
    img = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    img = cv2.resize(img, None,fx=gt_res, fy=gt_res, interpolation=cv2.INTER_LINEAR)
    # TODO: Experiment with thermal normalization
    return (img * (255 / img.max())).astype(np.uint8)

def preprocess_rgb(rgb_path, gt_res=0.5):
    if rgb_path is None:
        return None
    img = cv2.imread(rgb_path).astype(np.uint8)
    img = cv2.resize(img, None,fx=gt_res, fy=gt_res, interpolation=cv2.INTER_LINEAR)
    return img

def postprocess_gt(gt):
    gt = cv2.resize(gt, None,fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    return np.rot90(gt, k=3)

def get_thermal_mono_data(output_dir, main_dir, aligned_dir, sgbm_module: StereoBlockMatching):
    dirs = os.listdir(main_dir)
    for d in tqdm.tqdm(dirs):
        if not os.path.isdir(os.path.join(main_dir, d)):
            continue
        
        subdir = os.path.join(main_dir, f'{d}/rectified/')
        aligned_subdir = os.path.join(aligned_dir, f'{d}/')
        for nir_1 in tqdm.tqdm(glob.glob(f'{subdir}/p*m*l0*')):
            module, frame_num, _ = tuple(nir_1.split('_'))
            aligned_module = module.replace(subdir, aligned_subdir)
            nir_2 = load_single_file_with_pattern(module, frame_num, 3)
            rgb = preprocess_rgb(load_single_file_with_pattern(aligned_module, frame_num, 2))
            thermal = preprocess_thermal(load_single_file_with_pattern(aligned_module, frame_num, 1))

            if (nir_2 is None) or (rgb is None) or (thermal is None):
                continue

            gt = postprocess_gt(sgbm_module.predict(preprocess_nir(nir_1), preprocess_nir(nir_2)))
            base_name = module.split('/')[-1]
            np.save(f'{output_dir}/{base_name}_{frame_num}_rgb.npy', rgb)
            np.save(f'{output_dir}/{base_name}_{frame_num}_gt.npy', gt)
            np.save(f'{output_dir}/{base_name}_{frame_num}_thermal.npy', thermal)

import shutil

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect usage: Expected output directory as an argument only.")
        exit(0)

    if not os.path.exists(sys.argv[1]):
        print("Directory does not exist. Creating.")
        os.mkdir(sys.argv[1])
    else:
        print("Directory already exists. Removing all files.")
        shutil.rmtree(sys.argv[1])
        os.mkdir(sys.argv[1])
    sgbm = StereoBlockMatching(256)
    get_thermal_mono_data(sys.argv[1], THERMAL_CORD_MAIN_DIR, THERMAL_CORD_ALIGNED_DIR, sgbm)

