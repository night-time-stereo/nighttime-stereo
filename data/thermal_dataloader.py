import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

from config import *
from data.dataloader import DepthEstimationDataLoader


class ThermalDataloader(DepthEstimationDataLoader):
    def __init__(self):
        self.filepath = THERMAL_MONO_DIR
        super().__init__(self._load_dataset())

    def _load_dataset(self):
        outputs = []
        for i in sorted(glob.glob('%s/*gt*'%self.filepath)):
            sample = i.split('/')[-1]
            rgb = f'{self.filepath}/' + sample.replace('gt', 'rgb')
            thermal = f'{self.filepath}/' + sample.replace('gt', 'thermal')
            gt = i
            if os.path.exists(rgb) and os.path.exists(thermal):
                outputs.append((rgb, thermal, gt))
        return outputs

    def _preprocess(self, rgb, thermal, gt):
        rgb_img = np.load(rgb)
        thermal_img = np.load(thermal)
        gt_img = np.load(gt)
        return rgb_img, thermal_img, gt_img
