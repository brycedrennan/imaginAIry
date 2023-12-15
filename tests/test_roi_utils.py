import itertools
import random

from imaginairy.utils.roi_utils import (
    RoiNotInBoundsError,
    resize_roi_coordinates,
    square_roi_coordinate,
)


def test_square_roi_coordinate():
    img_sizes = (10, 100, 200, 511, 513, 1024)
    # iterate through all permutations of image sizes using itertools.product
    for img_width, img_height in itertools.product(img_sizes, img_sizes):
        # randomly generate a region of interest
        for _ in range(100):
            x1 = random.randint(0, img_width)
            y1 = random.randint(0, img_height)
            x2 = random.randint(x1, img_width)
            y2 = random.randint(y1, img_height)
            roi = x1, y1, x2, y2
            try:
                x1, y1, x2, y2 = square_roi_coordinate(roi, img_width, img_height)
            except RoiNotInBoundsError:
                continue
            assert (
                x2 - x1 == y2 - y1
            ), f"ROI is not square: img_width: {img_width}, img_height: {img_height}, roi: {roi}"


# resize_roi_coordinates


def test_square_resize_roi_coordinates():
    img_sizes = (10, 100, 200, 403, 511, 513, 604, 1024)
    # iterate through all permutations of image sizes using itertools.product
    img_sizes = list(itertools.product(img_sizes, img_sizes))

    for img_width, img_height in img_sizes:
        # randomly generate a region of interest
        rois = []
        for _ in range(100):
            x1 = random.randint(0 + 1, img_width - 1)
            y1 = random.randint(0 + 1, img_height - 1)
            x2 = random.randint(x1 + 1, img_width)
            y2 = random.randint(y1 + 1, img_height)
            roi = x1, y1, x2, y2
            rois.append(roi)
        rois.append((392, 85, 695, 389))
        for roi in rois:
            try:
                squared_roi = square_roi_coordinate(roi, img_width, img_height)
            except RoiNotInBoundsError:
                continue
            for n in range(10):
                factor = 1.25 + 0.3 * n
                x1, y1, x2, y2 = resize_roi_coordinates(
                    squared_roi, factor, img_width, img_height
                )
                assert (
                    x2 - x1 == y2 - y1
                ), f"ROI is not square: img_width: {img_width}, img_height: {img_height}, roi: {roi}"

                x1, y1, x2, y2 = resize_roi_coordinates(
                    squared_roi, factor, img_width, img_height, expand_up=False
                )
                assert (
                    x2 - x1 == y2 - y1
                ), f"ROI is not square: img_width: {img_width}, img_height: {img_height}, roi: {roi}"
