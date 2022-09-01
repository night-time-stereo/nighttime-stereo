
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import torch
import os

from model.sgbm import StereoBlockMatching
from data.kitti_dataloader import get_kitti_files, preprocess_data

dataset = "kitti"
KITTI_DIR = "/mnt/aidtr/external/kitti/stereo_flow_sceneflow_2015/training/"
kitti_dl = get_kitti_files(KITTI_DIR)
max_disp = 128
sgbm = StereoBlockMatching(max_disp)

if not os.path.exists(f'outputs/sgbm/{dataset}'):
    os.makedirs(f'outputs/sgbm/{dataset}')

for i in tqdm.tqdm(range(len(kitti_dl))):
    left, right, gt = preprocess_data(*kitti_dl[i])
    pred = sgbm.predict(left, right)

    np.save(f'outputs/sgbm/{dataset}/{kitti_dl[i][0].split("/")[-1]}.npy', pred)

    # print(pred.shape, gt.shape)
    # for k, m in metrics.items():
    #     print(f'{k}: {m(torch.from_numpy(pred), torch.from_numpy(gt))}')
    # error_map = ((pred - gt) ** 2) * (gt > 0)
    # mean_err = error_map.sum() / (gt > 0).sum()
    # print(np.sqrt(error_map.max()), np.sqrt(error_map.min()), np.sqrt(mean_err))
    # plt.imshow(error_map, cmap='inferno')
    # plt.savefig(f'debug/error_{i}.png')
    # plt.close()
    # plt.imshow(pred, cmap='inferno', vmin=0, vmax=max_disp)
    # plt.savefig(f'debug/sgbm_{i}.png')
    # plt.close()
    # print(pred.max(), pred.min(), gt.max(), gt.min())
    # plt.imshow(gt, cmap='inferno', vmin=0, vmax=max_disp)
    # plt.savefig(f'debug/gt_{i}.png')
    # plt.close()
    # break


