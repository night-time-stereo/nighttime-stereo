
from this import d
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import torch
import os

from model.sgbm import StereoBlockMatching
from data.kitti_dataloader import get_kitti_files, preprocess_data
from data.canon_dataloder import get_canon_files, preprocess_canon_data


KITTI_DIR = "/mnt/aidtr/external/kitti/stereo_flow_sceneflow_2015/training/"
CANON_DIR = "/mnt/aidtr/members/vanshajc/data/canon_images/dark/"

dataset = "canon"

if dataset == "kitti":
    dl = get_kitti_files(KITTI_DIR)
    preprocessor = preprocess_data
elif dataset == "canon":
    dl = get_canon_files(CANON_DIR)
    preprocessor = preprocess_canon_data

max_disp = 256
sgbm = StereoBlockMatching(max_disp)

if not os.path.exists(f'outputs/sgbm/{dataset}'):
    os.makedirs(f'outputs/sgbm/{dataset}')

for i in tqdm.tqdm(range(len(dl))):
    left, right = preprocessor(*dl[i])
    pred = sgbm.predict(left, right)

    np.save(f'outputs/sgbm/{dataset}/{dl[i][0].split("/")[-1]}.npy', pred)
    plt.imshow(pred, cmap='inferno', vmin=0, vmax=max_disp)
    plt.savefig(f'debug/dark_sgbm_{dataset}_{i}.png')
    plt.close()
