import logging

logger = logging.getLogger(__name__)


def square_roi_coordinate(roi, max_width, max_height, best_effort=False):
    """Given a region of interest, returns a square region of interest."""
    x1, y1, x2, y2 = roi
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    roi_width = x2 - x1
    roi_height = y2 - y1
    if roi_width < roi_height:
        diff = roi_height - roi_width
        x1 -= int(round(diff / 2))
        x2 += roi_height - (x2 - x1)
    elif roi_height < roi_width:
        diff = roi_width - roi_height
        y1 -= int(round(diff / 2))
        y2 += roi_width - (y2 - y1)

    x1, y1, x2, y2 = move_roi_into_bounds(
        (x1, y1, x2, y2), max_width, max_height, force=best_effort
    )
    width = x2 - x1
    height = y2 - y1
    if not best_effort and width != height:
        msg = f"ROI is not square: {width}x{height}"
        raise RuntimeError(msg)
    return x1, y1, x2, y2


def resize_roi_coordinates(
    roi, expansion_factor, max_width, max_height, expand_up=True
):
    """
    Resize a region of interest while staying within the bounds.

    setting expand_up to False will prevent the ROI from expanding upwards, which is useful when
    expanding something like a face roi to capture more of the person instead of empty space above them.

    """
    x1, y1, x2, y2 = roi
    side_length_x = x2 - x1
    side_length_y = y2 - y1

    max_expansion_factor = min(max_height / side_length_y, max_width / side_length_x)
    expansion_factor = min(expansion_factor, max_expansion_factor)

    expansion_x = int(round(side_length_x * expansion_factor - side_length_x))
    expansion_x_a = int(round(expansion_x / 2))
    expansion_x_b = expansion_x - expansion_x_a
    x1 -= expansion_x_a
    x2 += expansion_x_b

    expansion_y = int(round(side_length_y * expansion_factor - side_length_y))
    if expand_up:
        expansion_y_a = int(round(expansion_y / 2))
        expansion_y_b = expansion_y - expansion_y_a
        y1 -= expansion_y_a
        y2 += expansion_y_b
    else:
        y2 += expansion_y

    x1, y1, x2, y2 = move_roi_into_bounds((x1, y1, x2, y2), max_width, max_height)

    return x1, y1, x2, y2


class RoiNotInBoundsError(ValueError):
    """Error raised when a ROI is not within the bounds of the image."""


def move_roi_into_bounds(roi, max_width, max_height, force=False):
    """Move a region of interest into the bounds of the image."""
    x1, y1, x2, y2 = roi

    # move the ROI within the image boundaries
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > max_width:
        x1 -= x2 - max_width
        x2 = max_width
    if y2 > max_height:
        y1 -= y2 - max_height
        y2 = max_height
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    # Force ROI to fit within image boundaries (sacrificing size and aspect ratio of ROI)
    if force:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(max_width, x2)
        y2 = min(max_height, y2)
    if x1 < 0 or y1 < 0 or x2 > max_width or y2 > max_height:
        roi_width = x2 - x1
        roi_height = y2 - y1
        msg = f"Not possible to fit ROI into boundaries: {roi_width}x{roi_height} won't fit inside {max_width}x{max_height}"
        raise RoiNotInBoundsError(msg)

    return x1, y1, x2, y2
