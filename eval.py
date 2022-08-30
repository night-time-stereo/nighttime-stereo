
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt

from thermal.sgbm import StereoBlockMatching
from data.kitti_dataloader import get_kitti_files, preprocess_data


KITTI_DIR = "/mnt/aidtr/external/kitti/stereo_flow_sceneflow_2015/training/"
kitti_dl = get_kitti_files(KITTI_DIR)
max_disp = 128
sgbm = StereoBlockMatching(max_disp)

for i in tqdm.tqdm(range(len(kitti_dl))):
    print(kitti_dl[i])
    left, right, gt = preprocess_data(*kitti_dl[i])
    pred = sgbm.predict(left, right)
    plt.imshow(pred, cmap='inferno', vmin=0, vmax=max_disp)
    plt.savefig(f'debug/sgbm_{i}.png')
    plt.close()
    print(pred.max(), pred.min(), gt.max(), gt.min())
    plt.imshow(gt, cmap='inferno', vmin=0, vmax=max_disp)
    plt.savefig(f'debug/gt_{i}.png')
    plt.close()
    cv2.imwrite(f'debug/left_{i}.png', left)
    cv2.imwrite(f'debug/right_{i}.png', right)

