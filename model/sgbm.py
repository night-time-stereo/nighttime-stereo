from turtle import left
import cv2
import numpy as np


class StereoBlockMatching:
    """
    Reference: http://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=67877bd6fcc43163421fa0108c7df83bbc69fea3
    """
    def __init__(self, max_disp) -> None:
        self.max_disp = max_disp
        self.model = cv2.StereoSGBM_create(minDisparity=0,
                                           numDisparities=max_disp,
                                           blockSize=3,
                                           preFilterCap=63,
                                           speckleWindowSize=100,
                                           P1=3*3*4,
                                           P2=3*3*32,
                                           uniquenessRatio=10,
                                           speckleRange=32,
                                           disp12MaxDiff=1)

    def predict(self, left_img, right_img):
        assert 1 < len(left_img.shape) <= 3 
        assert left_img.shape == right_img.shape

        if len(left_img.shape) == 3: # batch
            batch_size = left_img.shape[0]
            pred_disp = [self.model.compute(left_img[i], right_img[i]).astype(np.float32) for i in range(batch_size)]
        else:
            pred_disp = self.model.compute(left_img, right_img).astype(np.float32) / 16
        return pred_disp
