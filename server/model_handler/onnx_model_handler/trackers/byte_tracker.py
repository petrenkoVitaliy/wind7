import numpy as np
from scipy.optimize import linear_sum_assignment

from server.utils import tprint


def box_diou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = np.array(boxes1, dtype=np.float32)
    b2 = np.array(boxes2, dtype=np.float32)

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / np.clip(union, a_min=1e-6, a_max=None)

    c1 = (b1[:, :2] + b1[:, 2:]) / 2
    c2 = (b2[:, :2] + b2[:, 2:]) / 2
    c_dist = ((c1[:, None, :] - c2[None, :, :]) ** 2).sum(axis=-1)

    elt = np.minimum(b1[:, None, :2], b2[None, :, :2])
    erb = np.maximum(b1[:, None, 2:], b2[None, :, 2:])
    ewh = np.clip(erb - elt, a_min=0, a_max=None)
    e_dist = (ewh ** 2).sum(axis=-1)

    diou = iou - (c_dist / np.clip(e_dist, a_min=1e-6, a_max=None))
    return diou


class ByteTrackBox:
    def __init__(self, obj_id, box):
        self.id = obj_id
        self.disappeared = 0

        self.x = np.zeros((8, 1), dtype=np.float32)
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        w, h = box[2] - box[0], box[3] - box[1]
        h = max(h, 1.0)

        self.x[:4, 0] = [cx, cy, w, h]

        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0

        self.H = np.eye(4, 8, dtype=np.float32)

        self._std_weight_pos = 1. / 10
        self._std_weight_vel = 1. / 10

        std_pos = self._std_weight_pos * h
        std_vel = self._std_weight_vel * h

        self.P = np.diag([
            2 * std_pos, 2 * std_pos, 2 * std_pos, 2 * std_pos,
            10 * std_vel, 10 * std_vel, 10 * std_vel, 10 * std_vel
        ]) ** 2

        self.predicted_box = box

    def predict(self):
        h = max(self.x[3, 0], 1.0)
        std_pos = self._std_weight_pos * h
        std_vel = self._std_weight_vel * h

        Q = np.diag([
            std_pos, std_pos, std_pos, std_pos,
            std_vel, std_vel, std_vel, std_vel
        ]) ** 2

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        cx, cy, w, h = self.x[:4, 0]
        self.predicted_box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        return self.predicted_box

    def correct(self, box):
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        w, h = box[2] - box[0], box[3] - box[1]
        h = max(h, 1.0)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        std_pos = self._std_weight_pos * h
        R = np.diag([std_pos, std_pos, std_pos, std_pos]) ** 2

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        self.disappeared = 0


class ByteTrack:
    def __init__(self, track_high_thresh=0.2, track_low_thresh=0.1, new_track_thresh=0.5, match_thresh=1.5, max_disappeared=30):
        tprint("INIT: ByteTrack initialized")
        self.next_object_id = 0
        self.tracks = {}

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        self.match_thresh = match_thresh
        self.max_disappeared = max_disappeared

    def update(self, boxes, scores, _frame=None):
        result_ids = [-1] * len(boxes)
        if len(boxes) == 0:
            for obj_id in list(self.tracks.keys()):
                self.tracks[obj_id].disappeared += 1
                if self.tracks[obj_id].disappeared > self.max_disappeared:
                    del self.tracks[obj_id]
            return result_ids

        for track in self.tracks.values():
            track.predict()

        high_indices = [i for i, s in enumerate(
            scores) if s >= self.track_high_thresh]
        low_indices = [i for i, s in enumerate(
            scores) if self.track_low_thresh <= s < self.track_high_thresh]

        high_boxes = [boxes[i] for i in high_indices]
        low_boxes = [boxes[i] for i in low_indices]

        track_ids_list = list(self.tracks.keys())
        predicted_boxes = [t.predicted_box for t in self.tracks.values()]

        unmatched_tracks = set(track_ids_list)
        unmatched_high_det = set(range(len(high_boxes)))

        if len(predicted_boxes) > 0 and len(high_boxes) > 0:
            diou_matrix = box_diou_matrix(predicted_boxes, high_boxes)
            cost_matrix = 1.0 - diou_matrix

            rows, cols = linear_sum_assignment(cost_matrix)

            for r, c in zip(rows, cols):
                if cost_matrix[r, c] <= self.match_thresh:
                    obj_id = track_ids_list[r]
                    orig_idx = high_indices[c]

                    self.tracks[obj_id].correct(boxes[orig_idx])
                    result_ids[orig_idx] = obj_id

                    unmatched_tracks.discard(obj_id)
                    unmatched_high_det.discard(c)

        unmatched_tracks_list = list(unmatched_tracks)
        remaining_predicted_boxes = [
            self.tracks[tid].predicted_box for tid in unmatched_tracks_list]

        if len(remaining_predicted_boxes) > 0 and len(low_boxes) > 0:
            diou_matrix_low = box_diou_matrix(
                remaining_predicted_boxes, low_boxes)
            cost_matrix_low = 1.0 - diou_matrix_low

            rows, cols = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(rows, cols):
                if cost_matrix_low[r, c] <= 1.0:
                    obj_id = unmatched_tracks_list[r]
                    orig_idx = low_indices[c]

                    self.tracks[obj_id].correct(boxes[orig_idx])
                    result_ids[orig_idx] = obj_id
                    unmatched_tracks.discard(obj_id)

        for tid in unmatched_tracks:
            self.tracks[tid].disappeared += 1
            if self.tracks[tid].disappeared > self.max_disappeared:
                del self.tracks[tid]

        for c in unmatched_high_det:
            orig_idx = high_indices[c]

            if scores[orig_idx] >= self.new_track_thresh:
                new_id = self.next_object_id
                self.tracks[new_id] = ByteTrackBox(new_id, boxes[orig_idx])
                result_ids[orig_idx] = new_id
                self.next_object_id += 1

        return result_ids
