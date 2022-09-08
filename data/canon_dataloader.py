import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

from config import *
from data.dataloader import DepthEstimationDataLoader


class CanonDataloader(DepthEstimationDataLoader):
    def __init__(self, mode):
        self.filepath = CANON_DARK_DIR if mode == "dark" else CANON_LIGHT_DIR
        super().__init__(self._load_dataset())

    def _load_dataset(self):
        outputs = []
        for i in sorted(glob.glob('%s/left*'%self.filepath)):
            l = f'{self.filepath}/' + i.split('/')[-1].split('\\')[-1]
            r = f'{self.filepath}/' + i.split('/')[-1].split('\\')[-1].replace('left', 'right')
            if os.path.exists(l) and os.path.exists(r):
                outputs.append((l, r))
        return outputs

    def _preprocess(self, left, right):
        left_img = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2GRAY)
        right_img = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2GRAY)
        return left_img, right_img, None
