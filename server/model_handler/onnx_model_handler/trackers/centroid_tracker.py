from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.spatial import distance


class CentroidTracker:
    next_object_id: int
    objects: OrderedDict[int, np.ndarray]
    disappeared: OrderedDict[int, int]
    max_disappeared: int
    max_distance: float

    def __init__(self, max_disappeared: int = 30, max_distance: float = 100) -> None:
        self.next_object_id: int = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_disappeared: int = max_disappeared
        self.max_distance: float = max_distance

    def update(self, boxes: list[list[float]]) -> list[int]:
        if len(boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return []

        input_centroids = np.zeros((len(boxes), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            input_centroids[i] = (
                int((startX + endX) / 2.0),
                int((startY + endY) / 2.0),
            )

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = distance.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows: set[int] = set()
            used_cols: set[int] = set()

            for row, col in zip(rows, cols, strict=True):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        result_ids: list[int] = []
        for i in range(len(input_centroids)):
            matched_id = -1
            for obj_id, centroid in self.objects.items():
                if np.array_equal(centroid, input_centroids[i]):
                    matched_id = obj_id
                    break
            result_ids.append(matched_id)

        return result_ids

    def register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]
