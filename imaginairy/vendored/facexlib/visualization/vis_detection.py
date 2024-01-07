import cv2
import numpy as np


def visualize_detection(img, bboxes_and_landmarks, save_path=None, to_bgr=False):
    """Visualize detection results.

    Args:
        img (Numpy array): Input image. CHW, BGR, [0, 255], uint8.
    """
    img = np.copy(img)
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for b in bboxes_and_landmarks:
        # confidence
        cv2.putText(img, f'{b[4]:.4f}', (int(b[0]), int(b[1] + 12)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # bounding boxes
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        # landmarks (for retinaface)
        cv2.circle(img, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img, (b[13], b[14]), 1, (255, 0, 0), 4)
    # save img
    if save_path is not None:
        cv2.imwrite(save_path, img)
