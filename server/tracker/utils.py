from __future__ import annotations

import numpy as np


def box_diou_matrix(boxes1: list[list[float]], boxes2: list[list[float]]) -> np.ndarray:
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
    e_dist = (ewh**2).sum(axis=-1)

    diou: np.ndarray = iou - (c_dist / np.clip(e_dist, a_min=1e-6, a_max=None))
    return diou
