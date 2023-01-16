import cv2

from imaginairy.img_utils import pillow_img_to_opencv_img


def calculate_blurriness_level(img):
    img = pillow_img_to_opencv_img(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = max(sharpness, 0.000001)
    bluriness = 1 / sharpness
    return bluriness


def is_blurry(img, threshold=0.91):
    return calculate_blurriness_level(img) > threshold
