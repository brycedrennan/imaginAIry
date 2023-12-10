import time

import numpy as np
from PIL import Image


def assert_image_similar_to_expectation(img, img_path, threshold=100):
    img.save(img_path)
    expected_img_path = img_path.replace("/test_output/", "/expected_output/")
    expected_img = Image.open(expected_img_path)
    norm_sum_sq_diff = calc_norm_sum_sq_diff(img, expected_img)

    if norm_sum_sq_diff > threshold:
        diff_img = Image.fromarray(np.asarray(img) - np.asarray(expected_img))
        diff_img.save(img_path + f"_diff_{norm_sum_sq_diff:.1f}.png")
        expected_img.save(img_path + "_expected.png")
        assert (
            norm_sum_sq_diff < threshold
        ), f"{norm_sum_sq_diff:.3f} is bigger than threshold {threshold}"


def calc_norm_sum_sq_diff(img, img2):
    sum_sq_diff = np.sum(
        (np.asarray(img).astype("float") - np.asarray(img2).astype("float")) ** 2
    )
    norm_sum_sq_diff = sum_sq_diff / np.sqrt(sum_sq_diff)
    return norm_sum_sq_diff


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None
        self.elapsed = None
        self.end = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

        print(f"{self.name} took {self.elapsed*1000:.2f} ms")
