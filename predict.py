
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

"""
Dataset options:
    - kitti
    - canon_dark
    - canon_light
    - thermal
"""
dataset = "canon_dark"

max_disp = 256
sgbm = StereoBlockMatching(max_disp)

if dataset == "kitti":
    dl = KittiDataloader()
elif "canon" in dataset:
    dl = CanonDataloader(dataset.split("_")[-1])
elif dataset == "thermal":
    # dl = get_thermal_mono_data(THERMAL_CORD_MAIN_DIR, THERMAL_CORD_ALIGNED_DIR, sgbm)
    exit(0)

if not os.path.exists(f'outputs/sgbm/{dataset}'):
    os.makedirs(f'outputs/sgbm/{dataset}')

for i in tqdm.tqdm(range(len(dl))):
    left, right = dl[i]
    pred = sgbm.predict(left[1], right[1])

    np.save(f'outputs/sgbm/{dataset}/{left[0].split("/")[-1]}.npy', pred)
    plt.imshow(pred, cmap='inferno', vmin=0, vmax=max_disp)
    plt.savefig(f'debug/sgbm_{dataset}_{i}.png')
    plt.close()
