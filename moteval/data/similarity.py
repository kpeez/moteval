"""Geometry similarity. For boxes: Jaccard IoU on xywh rows.

Mirrors TrackEval's box IoU (convert xywh -> corners, clamp negatives, guard the
zero-union case to 0) so later metric ports stay numerically comparable.
"""

import numpy as np


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of xywh boxes.

    ``boxes_a`` is ``(n, 4)``, ``boxes_b`` is ``(m, 4)``; returns ``(n, m)``.
    """
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    a = boxes_a.astype(np.float64)
    b = boxes_b.astype(np.float64)
    ax1, ay1, aw, ah = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)

    area_a = (aw * ah)[:, None]
    area_b = (bw * bh)[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def box_ioa(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Intersection-over-area matrix, normalised by each ``boxes_a`` box's area.

    ``boxes_a`` is ``(n, 4)`` xywh, ``boxes_b`` is ``(m, 4)`` xywh; returns
    ``(n, m)``. Mirrors TrackEval's ``_calculate_box_ious(..., do_ioa=True)`` used
    to test predictions against crowd ignore regions.
    """
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    a = boxes_a.astype(np.float64)
    b = boxes_b.astype(np.float64)
    ax1, ay1, aw, ah = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)

    area_a = (aw * ah)[:, None]
    return np.where(area_a > 0.0, inter / area_a, 0.0)
