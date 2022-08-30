import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt


def get_kitti_files(filepath):
    outputs = []

    for i in sorted(glob.glob('%s/image_2/*'%filepath)):
        l = f'{filepath}/image_2/' + i.split('/')[-1].split('\\')[-1]
        d = f'{filepath}/disp_occ_0/' + i.split('/')[-1].split('\\')[-1]
        r = f'{filepath}/image_3/' + i.split('/')[-1].split('\\')[-1]
        if os.path.exists(l) and os.path.exists(d):
            outputs.append((l, r, d))
    return outputs


def preprocess_data(left_path, right_path, gt_path):
    left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2GRAY)
    gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH) / 256.0
    return left_img, right_img, gt
