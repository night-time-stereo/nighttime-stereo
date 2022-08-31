
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import torch

from model.sgbm import StereoBlockMatching
from data.kitti_dataloader import get_kitti_files, preprocess_data
from metrics.kitti_metrics import BadX, Rmse

KITTI_DIR = "/mnt/aidtr/external/kitti/stereo_flow_sceneflow_2015/training/"
kitti_dl = get_kitti_files(KITTI_DIR)
max_disp = 128
sgbm = StereoBlockMatching(max_disp)

metrics = {'Bad-1 %': BadX(1.), 'Bad-4 %': BadX(4.), 'Rmse': Rmse()}

for i in tqdm.tqdm(range(len(kitti_dl))):
    left, right, gt = preprocess_data(*kitti_dl[i])
    pred = sgbm.predict(left, right)
    print(pred.shape, gt.shape)
    for k, m in metrics.items():
        print(f'{k}: {m(torch.from_numpy(pred), torch.from_numpy(gt))}')


print("Final Results")
for k, m in metrics.items():
        print(f"{k}: {m.compute()}")





    # plt.imshow(pred, cmap='inferno', vmin=0, vmax=max_disp)
    # plt.savefig(f'debug/sgbm_{i}.png')
    # plt.close()
    # print(pred.max(), pred.min(), gt.max(), gt.min())
    # plt.imshow(gt, cmap='inferno', vmin=0, vmax=max_disp)
    # plt.savefig(f'debug/gt_{i}.png')
    # plt.close()

