from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from server.tracker.tracker import Tracker
from server.tracker.utils import box_diou_matrix
from server.utils.formatter import L, tprint


class GMC:
    prev_gray: np.ndarray | None
    prev_points: np.ndarray | None
    downscale: int

    def __init__(self, downscale: int = 2) -> None:
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.downscale: int = downscale

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3
            )
            return np.eye(2, 3, dtype=np.float32)

        if self.prev_points is None or len(self.prev_points) < 4:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3
            )
            return np.eye(2, 3, dtype=np.float32)

        next_pts = np.empty_like(self.prev_points)
        matched_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, next_pts
        )

        good_old = self.prev_points[status == 1]
        good_new = matched_points[status == 1]

        if len(good_new) >= 4:
            warp_matrix, _ = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC
            )
            if warp_matrix is None:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        self.prev_gray = gray
        self.prev_points = cv2.goodFeaturesToTrack(
            gray, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3
        )

        assert warp_matrix is not None
        return warp_matrix


class BoTSORTBox:
    id: int
    disappeared: int
    x: np.ndarray
    F: np.ndarray
    H: np.ndarray
    _std_weight_pos: float
    _std_weight_vel: float
    P: np.ndarray
    predicted_box: list[float]

    def __init__(self, obj_id: int, box: list[float]) -> None:
        self.id: int = obj_id
        self.disappeared: int = 0

        self.x: np.ndarray = np.zeros((8, 1), dtype=np.float32)
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        w, h = box[2] - box[0], box[3] - box[1]
        h = max(h, 1.0)

        self.x[:4, 0] = [cx, cy, w, h]

        self.F: np.ndarray = np.eye(8, dtype=np.float32)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0
        self.H: np.ndarray = np.eye(4, 8, dtype=np.float32)

        self._std_weight_pos: float = 1.0 / 10
        self._std_weight_vel: float = 1.0 / 10

        std_pos = self._std_weight_pos * h
        std_vel = self._std_weight_vel * h

        self.P: np.ndarray = (
            np.diag(
                [
                    2 * std_pos,
                    2 * std_pos,
                    2 * std_pos,
                    2 * std_pos,
                    10 * std_vel,
                    10 * std_vel,
                    10 * std_vel,
                    10 * std_vel,
                ]
            )
            ** 2
        )

        self.predicted_box: list[float] = box

    def apply_camera_motion(self, warp_matrix: np.ndarray) -> None:
        R = warp_matrix[:, :2]
        T = warp_matrix[:, 2]

        pos = self.x[:2].T
        new_pos = pos @ R.T + T
        self.x[:2] = new_pos.T

        scale = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        self.x[2:4] *= scale

        vel = self.x[4:6].T
        new_vel = vel @ R.T
        self.x[4:6] = new_vel.T

    def predict(self) -> list[float]:
        h = max(self.x[3, 0], 1.0)
        std_pos = self._std_weight_pos * h
        std_vel = self._std_weight_vel * h

        Q: np.ndarray = (
            np.diag(
                [std_pos, std_pos, std_pos, std_pos, std_vel, std_vel, std_vel, std_vel]
            )
            ** 2
        )

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        cx, cy, w, h = self.x[:4, 0]
        self.predicted_box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        return self.predicted_box

    def correct(self, box: list[float]) -> None:
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        w, h = box[2] - box[0], box[3] - box[1]
        h = max(h, 1.0)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        std_pos = self._std_weight_pos * h
        R = np.diag([std_pos, std_pos, std_pos, std_pos]) ** 2

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = np.linalg.solve(S, self.H @ self.P).T

        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        self.disappeared = 0


class BoTSORT(Tracker):
    next_object_id: int
    tracks: dict[int, BoTSORTBox]
    gmc: GMC
    _gmc_cnt: int
    gmc_skip: int
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    match_thresh: float
    max_disappeared: int

    def __init__(
        self,
        track_high_thresh: float = 0.2,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.5,
        match_thresh: float = 1.5,
        max_disappeared: int = 30,
    ) -> None:
        tprint(L.INIT_BOTSORT)
        self.next_object_id: int = 0
        self.tracks: dict[int, BoTSORTBox] = {}
        self.gmc: GMC = GMC(downscale=2)
        self._gmc_cnt: int = 0
        self.gmc_skip: int = 4

        self.track_high_thresh: float = track_high_thresh
        self.track_low_thresh: float = track_low_thresh
        self.new_track_thresh: float = new_track_thresh
        self.match_thresh: float = match_thresh
        self.max_disappeared: int = max_disappeared

    def update(
        self, boxes: list[list[float]], scores: list[float], frame: Any = None
    ) -> list[int]:
        result_ids = [-1] * len(boxes)

        if frame is not None and len(self.tracks) > 0:
            self._gmc_cnt = (self._gmc_cnt + 1) % self.gmc_skip
            if self._gmc_cnt == 0:
                warp_matrix = self.gmc.apply(frame)
                for track in self.tracks.values():
                    track.apply_camera_motion(warp_matrix)

        if len(boxes) == 0:
            for obj_id in list(self.tracks.keys()):
                self.tracks[obj_id].disappeared += 1
                if self.tracks[obj_id].disappeared > self.max_disappeared:
                    del self.tracks[obj_id]
            return result_ids

        for track in self.tracks.values():
            track.predict()

        high_indices = [i for i, s in enumerate(scores) if s >= self.track_high_thresh]
        low_indices = [
            i
            for i, s in enumerate(scores)
            if self.track_low_thresh <= s < self.track_high_thresh
        ]

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

            for r, c in zip(rows, cols, strict=True):
                if cost_matrix[r, c] <= self.match_thresh:
                    obj_id = track_ids_list[r]
                    orig_idx = high_indices[c]

                    self.tracks[obj_id].correct(boxes[orig_idx])
                    result_ids[orig_idx] = obj_id

                    unmatched_tracks.discard(obj_id)
                    unmatched_high_det.discard(c)

        unmatched_tracks_list = list(unmatched_tracks)
        remaining_predicted_boxes = [
            self.tracks[tid].predicted_box for tid in unmatched_tracks_list
        ]

        if len(remaining_predicted_boxes) > 0 and len(low_boxes) > 0:
            diou_matrix_low = box_diou_matrix(remaining_predicted_boxes, low_boxes)
            cost_matrix_low = 1.0 - diou_matrix_low

            rows, cols = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(rows, cols, strict=True):
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
                self.tracks[new_id] = BoTSORTBox(new_id, boxes[orig_idx])
                result_ids[orig_idx] = new_id
                self.next_object_id += 1

        return result_ids
