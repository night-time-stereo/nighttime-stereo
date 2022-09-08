
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import torch

from data.kitti_dataloader import KittiDataloader
from metrics.kitti_metrics import BadX, Rmse
from config import *

metrics = {'Bad-1 %': BadX(1.), 'Bad-4 %': BadX(4.), 'Rmse': Rmse()}

method = "hsm"
dataset = "kitti"
predicted_dir = f'outputs/{method}/{dataset}'
verbose = False

dl = KittiDataloader()

for i in tqdm.tqdm(range(len(dl))):
    left, right, gt = dl[i]
    pred = np.load(f'{predicted_dir}/{left[0].split("/")[-1]}.npy')
    assert pred.shape == gt.shape
    for k, m in metrics.items():
        val = m(torch.from_numpy(pred), torch.from_numpy(gt))
        if verbose:
            print(f'{k}: {val}')


print("Final Results")
for k, m in metrics.items():
        print(f"{k}: {m.compute()}")
