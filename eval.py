
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import torch

from data.kitti_dataloader import get_kitti_files, preprocess_data
from metrics.kitti_metrics import BadX, Rmse

KITTI_DIR = "/mnt/aidtr/external/kitti/stereo_flow_sceneflow_2015/training/"
kitti_dl = get_kitti_files(KITTI_DIR)
metrics = {'Bad-1 %': BadX(1.), 'Bad-4 %': BadX(4.), 'Rmse': Rmse()}

method = "hsm"
dataset = ""
predicted_dir = f'outputs/{method}/{dataset}'
verbose = False

for i in tqdm.tqdm(range(len(kitti_dl))):
    left, right, gt = preprocess_data(*kitti_dl[i])
    pred = np.load(f'{predicted_dir}/{kitti_dl[i][0].split("/")[-1]}.npy')
    assert pred.shape == gt.shape
    for k, m in metrics.items():
        val = m(torch.from_numpy(pred), torch.from_numpy(gt))
        if verbose:
            print(f'{k}: {val}')


print("Final Results")
for k, m in metrics.items():
        print(f"{k}: {m.compute()}")
