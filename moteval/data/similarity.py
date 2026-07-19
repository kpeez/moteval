"""Geometry similarity. Boxes: Jaccard IoU on xywh rows. Masks: pycocotools RLE IoU.

Box IoU mirrors TrackEval's (convert xywh -> corners, clamp negatives, guard the
zero-union case to 0). Mask IoU/IoA call ``pycocotools.mask.iou`` exactly as
TrackEval's ``_calculate_mask_ious`` does — the ``iscrowd`` flag switches the
denominator to intersection-over-area for ignore-region tests — so metric ports
stay numerically comparable. RLE encoding always goes through Fortran-order
arrays, matching the pycocotools contract.
"""

import numpy as np
from pycocotools import mask as mask_utils


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

    # areas derived from the converted corners, exactly as upstream:
    # ((x0+w)-x0) differs from w by an ULP when x0+w rounds, and LocA's
    # similarity sums are bit-sensitive to that difference.
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    eps = np.finfo("float").eps
    inter[area_a <= eps, :] = 0.0
    inter[:, area_b <= eps] = 0.0
    inter[union <= eps] = 0.0
    union[union <= eps] = 1.0
    return inter / union


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

    # corner-derived area + eps guard, exactly as upstream's do_ioa branch
    area_a = (ax2 - ax1) * (ay2 - ay1)
    ioas = np.zeros_like(inter)
    valid = area_a > np.finfo("float").eps
    ioas[valid, :] = inter[valid, :] / area_a[valid][:, None]
    return ioas


def mask_iou(masks_a: list, masks_b: list) -> np.ndarray:
    """IoU matrix between two lists of RLE masks, shaped ``(len(a), len(b))``.

    Mirrors TrackEval ``_calculate_mask_ious(..., is_encoded=True, do_ioa=False)``:
    ``pycocotools.mask.iou`` with every ``iscrowd`` flag False, reshaping the
    empty-input ``[]`` return to a properly-shaped zero-size array.
    """
    ious = mask_utils.iou(masks_a, masks_b, [False] * len(masks_b))
    if len(masks_a) == 0 or len(masks_b) == 0:
        ious = np.asarray(ious).reshape(len(masks_a), len(masks_b))
    return ious


def mask_ioa(masks_a: list, masks_b: list) -> np.ndarray:
    """Intersection-over-area matrix, normalised by each ``masks_a`` mask's area.

    Mirrors TrackEval ``_calculate_mask_ious(..., is_encoded=True, do_ioa=True)``,
    used to test unmatched predictions against the merged crowd-ignore mask:
    every ``masks_b`` entry carries ``iscrowd=True`` so the denominator is the
    ``masks_a`` mask's own area.
    """
    ioas = mask_utils.iou(masks_a, masks_b, [True] * len(masks_b))
    if len(masks_a) == 0 or len(masks_b) == 0:
        ioas = np.asarray(ioas).reshape(len(masks_a), len(masks_b))
    return ioas


def encode_mask(mask: np.ndarray) -> dict:
    """Encode one binary ``(h, w)`` mask as a compressed RLE dict.

    pycocotools requires Fortran-contiguous uint8 input; C-order arrays are
    converted, never rejected, so callers can pass masks built naturally in
    C order.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a single (h, w) mask, got shape {mask.shape}")
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def decode_mask(rle: dict) -> np.ndarray:
    """Decode one RLE dict back to a binary ``(h, w)`` uint8 mask."""
    return mask_utils.decode(rle)


def merge_masks(rles: list) -> dict:
    """Union a list of RLE masks into one RLE (empty list -> empty mask), exactly
    as TrackEval merges per-frame ignore regions: ``mask.merge(intersect=False)``."""
    return mask_utils.merge(rles, intersect=False)


def masks_overlap(rles: list) -> bool:
    """Whether any two RLE masks in the list overlap (TrackEval rejects such input)."""
    merged_area = mask_utils.area(mask_utils.merge(rles, intersect=False))
    total_area = sum(mask_utils.area(rle) for rle in rles)
    return bool(total_area > merged_area)
