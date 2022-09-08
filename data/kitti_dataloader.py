import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

from config import *
from data.dataloader import DepthEstimationDataLoader


class KittiDataloader(DepthEstimationDataLoader):
    def __init__(self):
        self.filepath = KITTI_DIR
        super().__init__(self._load_dataset())

    def _load_dataset(self):
        outputs = []
        for i in sorted(glob.glob('%s/image_2/*'%self.filepath)):
            sample = i.split('/')[-1].split('\\')[-1]
            l = f'{self.filepath}/image_2/' + sample
            d = f'{self.filepath}/disp_occ_0/' + sample
            r = f'{self.filepath}/image_3/' + sample
            if os.path.exists(l) and os.path.exists(d):
                outputs.append((l, r, d))
        return outputs

    def _preprocess(self, left_path, right_path, gt_path):
        left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2GRAY)
        right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2GRAY)
        gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH) / 256.0
        return left_img, right_img, gt
