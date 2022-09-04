import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt


def get_canon_files(filepath):
    outputs = []

    for i in sorted(glob.glob('%s/left*'%filepath)):
        l = f'{filepath}/' + i.split('/')[-1].split('\\')[-1]
        r = f'{filepath}/' + i.split('/')[-1].split('\\')[-1].replace('left', 'right')
        if os.path.exists(l) and os.path.exists(r):
            outputs.append((l, r))
    return outputs


def preprocess_canon_data(left_path, right_path):
    left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2GRAY)
    return left_img, right_img
