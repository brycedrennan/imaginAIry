import cv2
import numpy as np


def visualize_alignment(img, landmarks, save_path=None, to_bgr=False):
    img = np.copy(img)
    h, w = img.shape[0:2]
    circle_size = int(max(h, w) / 150)
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for landmarks_face in landmarks:
        for lm in landmarks_face:
            cv2.circle(img, (int(lm[0]), int(lm[1])), 1, (0, 150, 0), circle_size)

    # save img
    if save_path is not None:
        cv2.imwrite(save_path, img)
