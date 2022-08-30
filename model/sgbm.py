from turtle import left
import cv2
import numpy as np


class StereoBlockMatching:
    def __init__(self, max_disp) -> None:
        print(max_disp, max_disp / 16)
        self.max_disp = max_disp
        self.model = cv2.StereoSGBM_create(minDisparity=-(max_disp // 2 - 1), numDisparities=max_disp, blockSize=15)

    def predict(self, left_img, right_img):
        assert 1 < len(left_img.shape) <= 3 
        assert left_img.shape == right_img.shape

        if len(left_img.shape) == 3: # batch
            batch_size = left_img.shape[0]
            pred_disp = [self.model.compute(left_img[i], right_img[i]).astype(np.float32) for i in range(batch_size)]
        else:
            pred_disp = self.model.compute(left_img, right_img).astype(np.float32) / 16
            # TODO: Investigate whether to shift negative disparities
            # pred_disp = (pred_disp - pred_disp.min())
        return pred_disp
