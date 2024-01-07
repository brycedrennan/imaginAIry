import numpy as np

from imaginairy.vendored.facexlib.tracking.data_association import associate_detections_to_trackers
from imaginairy.vendored.facexlib.tracking.kalman_tracker import KalmanBoxTracker


class SORT(object):
    """SORT: A Simple, Online and Realtime Tracker.

    Ref: https://github.com/abewley/sort
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits  # 最小的连续命中, 只有满足的才会被返回
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, img_size, additional_attr, detect_interval):
        """This method must be called once for each frame even with
        empty detections.
        NOTE:as in practical realtime MOT, the detector doesn't run on every
        single frame.

        Args:
            dets (Numpy array): detections in the format
                [[x0,y0,x1,y1,score], [x0,y0,x1,y1,score], ...]

        Returns:
             a similar array, where the last column is the object ID.
        """
        self.frame_count += 1

        # get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []  # To be deleted
        ret = []
        # predict tracker position using Kalman filter
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()  # Kalman predict ,very fast ,<1ms
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if dets != []:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(  # noqa: E501
                dets, trks)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                    trk.face_attributes.append(additional_attr[d[0]])

            # create and initialize new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                trk.face_attributes.append(additional_attr[i])
                print(f'New tracker: {trk.id + 1}.')
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([])

            d = trk.get_state()
            # get return tracklet
            # 1) time_since_update < 1: detected
            # 2) i) hit_streak >= min_hits: 最小的连续命中
            #    ii) frame_count <= min_hits: 最开始的几帧
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracklet
            # 1) time_since_update >= max_age: 多久没有更新了
            # 2) predict_num: 连续预测的帧数
            # 3) out of image size
            if (trk.time_since_update >= self.max_age) or (trk.predict_num >= detect_interval) or (
                    d[2] < 0 or d[3] < 0 or d[0] > img_size[1] or d[1] > img_size[0]):
                print(f'Remove tracker: {trk.id + 1}')
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        else:
            return np.empty((0, 5))
