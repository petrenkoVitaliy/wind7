import time
import numpy as np
from scipy.spatial import distance


class SingleTrackNumPy:
    def __init__(self, obj_id, initial_centroid):
        self.id = obj_id
        self.disappeared = 0

        self.F = np.eye(4, dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.Q = np.eye(4, dtype=np.float32) * 0.1
        self.R = np.eye(2, dtype=np.float32) * 4.0

        self.P = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 100, 0],
            [0, 0, 0, 100]
        ], dtype=np.float32)

        self.x = np.array([
            [initial_centroid[0]],
            [initial_centroid[1]],
            [0.0],
            [0.0]
        ], dtype=np.float32)

        self.xpred = np.copy(self.x)
        self.Ppred = np.copy(self.P)
        self.predicted_centroid = initial_centroid

    def predict(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.xpred = self.F @ self.x
        self.Ppred = self.F @ self.P @ self.F.T + self.Q

        self.x = self.xpred
        self.P = self.Ppred

        self.predicted_centroid = (int(self.x[0, 0]), int(self.x[1, 0]))
        return self.predicted_centroid

    def correct(self, centroid):
        z = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        y = z - self.H @ self.xpred
        S = self.H @ self.Ppred @ self.H.T + self.R
        K = self.Ppred @ self.H.T @ np.linalg.inv(S)

        self.x = self.xpred + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.Ppred
        self.disappeared = 0


class KalmanTracker:
    def __init__(self, max_disappeared=30, max_distance=100, default_dt=0.05):
        self.next_object_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.default_dt = default_dt

        self.last_time = None

    def update(self, boxes):
        result_ids = [-1] * len(boxes)

        current_time = time.perf_counter()
        if self.last_time is None:
            dt = self.default_dt
        else:
            dt = current_time - self.last_time
            if dt <= 0 or dt > 1.0:
                dt = self.default_dt

        self.last_time = current_time

        for track in self.tracks.values():
            track.predict(dt)

        if len(boxes) == 0:
            for obj_id in list(self.tracks.keys()):
                self.tracks[obj_id].disappeared += 1
                if self.tracks[obj_id].disappeared > self.max_disappeared:
                    del self.tracks[obj_id]
            return result_ids

        input_centroids = np.zeros((len(boxes), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            input_centroids[i] = (int((startX + endX) / 2.0),
                                  int((startY + endY) / 2.0))

        if len(self.tracks) == 0:
            for i in range(len(input_centroids)):
                result_ids[i] = self.register(input_centroids[i])
        else:
            object_ids = list(self.tracks.keys())
            predicted_centroids = [
                t.predicted_centroid for t in self.tracks.values()]

            D = distance.cdist(np.array(predicted_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.tracks[object_id].correct(input_centroids[col])
                result_ids[col] = object_id

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.tracks[object_id].disappeared += 1
                if self.tracks[object_id].disappeared > self.max_disappeared:
                    del self.tracks[object_id]

            for col in unused_cols:
                result_ids[col] = self.register(input_centroids[col])

        return result_ids

    def register(self, centroid):
        self.tracks[self.next_object_id] = SingleTrackNumPy(
            self.next_object_id, centroid)
        obj_id = self.next_object_id
        self.next_object_id += 1
        return obj_id
