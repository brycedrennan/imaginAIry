"""
For each detected item, it computes the intersection over union (IOU) w.r.t.
each tracked object. (IOU matrix)
Then, it applies the Hungarian algorithm (via linear_assignment) to assign each
det. item to the best possible tracked item (i.e. to the one with max IOU)
"""

import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment as linear_assignment


@jit
def iou(bb_test, bb_gt):
    """Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) *
              (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    """Assigns detections to tracked object (both represented as bounding boxes)

    Returns:
        3 lists of matches, unmatched_detections and unmatched_trackers.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    # The linear assignment module tries to minimize the total assignment cost.
    # In our case we pass -iou_matrix as we want to maximise the total IOU
    # between track predictions and the frame detection.
    row_ind, col_ind = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in row_ind:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in col_ind:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for row, col in zip(row_ind, col_ind):
        if iou_matrix[row, col] < iou_threshold:
            unmatched_detections.append(row)
            unmatched_trackers.append(col)
        else:
            matches.append(np.array([[row, col]]))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
