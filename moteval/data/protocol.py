"""Declarative per-benchmark preprocessing `Protocol` + one shared engine.

A `Protocol` is a value object: it declares *what* preprocessing a benchmark
needs (evaluated classes, distractor classes, ignore-region IoA threshold,
GT conf-zero semantics, frame convention) but never *how* to run it. `preprocess_frame`
is the single engine that executes any Protocol on one frame, replicating
TrackEval ``get_preprocessed_seq_data`` semantics (commit 12c8791b) exactly:

    1. keep only predictions of the class under evaluation;
    2. Hungarian-match those predictions against *all* GT (distractor classes
       included) and drop predictions matched to a distractor-class GT box;
    3. drop unmatched predictions whose IoA with an ignore region exceeds the
       threshold;
    4. keep only GT of the evaluated class, dropping conf-zero rows.

The engine is geometry-agnostic: it consumes precomputed ``similarity`` (GT x pred)
and ``ignore_ioa`` (pred x ignore) matrices, so masks slot in behind the same call
by swapping how those matrices are built during conversion.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from moteval.data.model import FrameConvention

_EPS = float(np.finfo("float").eps)


@dataclass(frozen=True)
class Protocol:
    """Declarative preprocessing rules for one benchmark.

    ``eval_classes`` are the class ids scored (one `SequenceData` view per class).
    ``distractor_classes`` are matched against but never evaluated. Thresholds carry
    TrackEval's canonical ``0.5`` defaults; ``drop_zero_conf_gt`` toggles the
    zero-marked GT exclusion (off only for preproc-free benchmarks like MOT15).
    ``matching_fill`` is the score below-threshold pairs get during the Hungarian
    matching step: upstream's box path zeroes them while its MOTS path sets
    ``-10000``, and the two can tie-break differently, so the value is a declared
    protocol parameter replicated exactly.
    """

    name: str
    frame_convention: FrameConvention
    eval_classes: tuple[int, ...]
    distractor_classes: tuple[int, ...] = ()
    distractor_iou_threshold: float = 0.5
    ignore_iou_threshold: float = 0.5
    drop_zero_conf_gt: bool = True
    matching_fill: float = 0.0


@dataclass(frozen=True)
class RawFrame:
    """One frame of raw GT + prediction arrays plus precomputed geometry.

    ``similarity`` is ``(num_gt, num_pred)``; ``ignore_ioa`` is
    ``(num_pred, num_ignore)`` intersection-over-area of each prediction against
    each ignore region. Ids are raw (un-densified).
    """

    gt_ids: np.ndarray
    gt_classes: np.ndarray
    gt_conf: np.ndarray
    gt_dets: np.ndarray
    pred_ids: np.ndarray
    pred_classes: np.ndarray
    pred_confidences: np.ndarray
    pred_dets: np.ndarray
    similarity: np.ndarray
    ignore_ioa: np.ndarray


@dataclass(frozen=True)
class PreprocessedFrame:
    """One frame after preprocessing: only surviving GT + predictions remain."""

    gt_ids: np.ndarray
    gt_dets: np.ndarray
    pred_ids: np.ndarray
    pred_confidences: np.ndarray
    pred_dets: np.ndarray
    similarity: np.ndarray


def preprocess_frame(frame: RawFrame, protocol: Protocol, cls_id: int) -> PreprocessedFrame:
    """Run the shared preprocessing engine on a single frame for one class."""
    pred_keep = np.flatnonzero(frame.pred_classes == cls_id)
    pred_ids = frame.pred_ids[pred_keep]
    pred_confidences = frame.pred_confidences[pred_keep]
    pred_dets = frame.pred_dets[pred_keep]
    similarity = frame.similarity[:, pred_keep]
    ignore_ioa = frame.ignore_ioa[pred_keep]

    # Step 2: Hungarian-match predictions to all GT (distractors included); drop
    # predictions matched above threshold to a distractor-class GT box. The
    # `< thr - eps` fill (0 for boxes, -10000 for MOTS) and `> 0 + eps` guard
    # mirror TrackEval exactly.
    match_cols = np.array([], dtype=np.int64)
    to_remove_matched = np.array([], dtype=np.int64)
    if frame.gt_ids.shape[0] > 0 and pred_ids.shape[0] > 0:
        matching_scores = similarity.copy()
        matching_scores[matching_scores < protocol.distractor_iou_threshold - _EPS] = (
            protocol.matching_fill
        )
        match_rows, match_cols = linear_sum_assignment(-matching_scores)
        actually_matched = matching_scores[match_rows, match_cols] > 0 + _EPS
        match_rows = match_rows[actually_matched]
        match_cols = match_cols[actually_matched]
        is_distractor = np.isin(frame.gt_classes[match_rows], protocol.distractor_classes)
        to_remove_matched = match_cols[is_distractor]

    # Step 3: drop unmatched predictions sitting inside an ignore region.
    unmatched = np.delete(np.arange(pred_ids.shape[0]), match_cols)
    to_remove_unmatched = np.array([], dtype=np.int64)
    if ignore_ioa.shape[1] > 0 and unmatched.shape[0] > 0:
        within_ignore = np.any(ignore_ioa[unmatched] > protocol.ignore_iou_threshold + _EPS, axis=1)
        to_remove_unmatched = unmatched[within_ignore]

    to_remove = np.concatenate((to_remove_matched, to_remove_unmatched)).astype(np.int64)
    pred_ids = np.delete(pred_ids, to_remove, axis=0)
    pred_confidences = np.delete(pred_confidences, to_remove, axis=0)
    pred_dets = np.delete(pred_dets, to_remove, axis=0)
    similarity = np.delete(similarity, to_remove, axis=1)

    # Step 4: keep only evaluated-class GT, excluding conf-zero rows when declared.
    gt_keep = frame.gt_classes == cls_id
    if protocol.drop_zero_conf_gt:
        gt_keep = gt_keep & (frame.gt_conf != 0)
    return PreprocessedFrame(
        gt_ids=frame.gt_ids[gt_keep],
        gt_dets=frame.gt_dets[gt_keep],
        pred_ids=pred_ids,
        pred_confidences=pred_confidences,
        pred_dets=pred_dets,
        similarity=similarity[gt_keep],
    )
