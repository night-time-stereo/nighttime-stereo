
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import torch
import os
from config import *

from model.sgbm import StereoBlockMatching
from data.kitti_dataloader import KittiDataloader
from data.canon_dataloader import CanonDataloader

dataset = "canon"

max_disp = 256
sgbm = StereoBlockMatching(max_disp)

if dataset == "kitti":
    dl = KittiDataloader()
elif dataset == "canon":
    dl = CanonDataloader()
elif dataset == "thermal":
    # dl = get_thermal_mono_data(THERMAL_CORD_MAIN_DIR, THERMAL_CORD_ALIGNED_DIR, sgbm)
    exit(0)

if not os.path.exists(f'outputs/sgbm/{dataset}'):
    os.makedirs(f'outputs/sgbm/{dataset}')

for i in tqdm.tqdm(range(len(dl))):
    left, right = dl[i]
    pred = sgbm.predict(left, right)

    np.save(f'outputs/sgbm/{dataset}/{dl[i][0].split("/")[-1]}.npy', pred)
    plt.imshow(pred, cmap='inferno', vmin=0, vmax=max_disp)
    plt.savefig(f'debug/dark_sgbm_{dataset}_{i}.png')
    plt.close()
