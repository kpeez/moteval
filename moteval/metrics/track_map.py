"""TrackMAP metric: track-level average precision/recall over IoU thresholds,
computed separately over the whole track set and over area/time-length subsets.

Ports TrackEval's TAO-style algorithm from the ``data`` dict its datasets build (track
areas/lengths/scores precomputed as means over each track's own detections): per-track
box IoU (summed intersection/union over the union of frames either track appears in),
greedy per-IoU-threshold matching with gt sorted ignore-last, right-to-left monotonic
precision interpolation, and ``np.searchsorted``-based recall-threshold indexing -- so
per-field results are bit-identical to the oracle. Track-level views (frames, areas,
lengths, scores) are derived here from `SequenceData` alone (ADR-0002); no bundled
dataset precomputes them for moteval.

One fix to a non-numeric hazard (not a quirk to replicate, per CLAUDE.md's ID-densification
guidance): upstream tracks "is this gt already matched" via ``gt_m[thr, gt] > 0``, comparing
against the literal matched *track id* -- a latent bug when a valid id is ``0`` (silently
treated as "still unmatched"). Upstream datasets never emit id 0, so it never fires there;
`SequenceData` densifies ids starting at 0, so it would fire constantly here. We track
matched state with an explicit boolean array instead, which is what the check clearly
intends and never changes a result upstream would produce on non-degenerate (id != 0) input.

Documented divergence (the sole permitted one, see README's "Known divergences"): upstream's
``combine_classes_det_averaged`` is a copy-paste of the class-averaged combiner (never
actually weights by detections). moteval implements the intended behavior instead.
"""

import numpy as np

from moteval.data.model import SequenceData
from moteval.metrics.base import Metric, Scores

EPS = np.finfo(float).eps

IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)
RECALL_THRESHOLDS = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01) + 1), endpoint=True)

AREA_RANGES = ((0**2, 32**2), (32**2, 96**2), (96**2, 1e5**2))
AREA_RANGE_LABELS = ("area_s", "area_m", "area_l")
TIME_RANGES = ((0, 3), (3, 10), (10, 1e5))
TIME_RANGE_LABELS = ("time_s", "time_m", "time_l")
LABELS = ("all", *AREA_RANGE_LABELS, *TIME_RANGE_LABELS)

_AP_FIELDS = tuple(f"AP_{lbl}" for lbl in LABELS)
_AR_FIELDS = tuple(f"AR_{lbl}" for lbl in LABELS)
_WEIGHT_PREFIX = "_num_dt_"
_DT_MATCHES = "_dt_matches_"
_DT_IGNORE = "_dt_ignore_"
_GT_IGNORE = "_gt_ignore_"
_DT_SCORES = "_dt_scores"

_Tracks = tuple[list[dict[int, np.ndarray]], list[int], list[float], list[float] | None]


def _weight_field(lbl: str) -> str:
    return f"{_WEIGHT_PREFIX}{lbl}"


def _group_tracks(
    ids_per_frame: tuple[np.ndarray, ...],
    boxes_per_frame: tuple[np.ndarray, ...],
    num_ids: int,
    confidences_per_frame: tuple[np.ndarray, ...] | None = None,
) -> _Tracks:
    """Group frame-major arrays into per-track views, indexed ``0..num_ids-1``.

    Mirrors upstream's dataset-side track precomputation (``track['area']`` and, for
    detections, ``track['score']``): both are the mean over the track's own detections
    (TAO/BURST convention -- YouTubeVIS instead stores one score per annotation file
    entry, not applicable here since `SequenceData` only carries per-detection confidences).
    """
    frames: list[dict[int, np.ndarray]] = [{} for _ in range(num_ids)]
    conf_sums = np.zeros(num_ids) if confidences_per_frame is not None else None
    for t, ids in enumerate(ids_per_frame):
        boxes = boxes_per_frame[t]
        for i, track_id in enumerate(ids):
            frames[track_id][t] = boxes[i]
            if conf_sums is not None and confidences_per_frame is not None:
                conf_sums[track_id] += confidences_per_frame[t][i]
    lengths = [len(f) for f in frames]
    areas = [float(np.mean([box[2] * box[3] for box in f.values()])) if f else 0.0 for f in frames]
    scores = None
    if conf_sums is not None:
        scores = [conf_sums[i] / lengths[i] if lengths[i] else 0.0 for i in range(num_ids)]
    return frames, lengths, areas, scores


def _track_iou(dt_frames: dict[int, np.ndarray], gt_frames: dict[int, np.ndarray]) -> float:
    intersect = 0.0
    union = 0.0
    for t in set(gt_frames) | set(dt_frames):
        g = gt_frames.get(t)
        d = dt_frames.get(t)
        if d is not None and g is not None:
            dx, dy, dw, dh = d
            gx, gy, gw, gh = g
            w = max(min(dx + dw, gx + gw) - max(dx, gx), 0.0)
            h = max(min(dy + dh, gy + gh) - max(dy, gy), 0.0)
            i = w * h
            union += dw * dh + gw * gh - i
            intersect += i
        elif g is not None:
            union += g[2] * g[3]
        elif d is not None:
            union += d[2] * d[3]
    return intersect / union if union > 0 else 0.0


def _ignore_masks(num_ids: int, lengths: list[int], areas: list[float]) -> list[np.ndarray]:
    masks = [np.zeros(num_ids, dtype=int)]
    for lo, hi in AREA_RANGES:
        masks.append(np.array([0 if lo - EPS <= a <= hi + EPS else 1 for a in areas], dtype=int))
    for lo, hi in TIME_RANGES:
        masks.append(
            np.array([0 if lo - EPS <= length <= hi + EPS else 1 for length in lengths], dtype=int)
        )
    return masks


class TrackMAP(Metric):
    fields = _AP_FIELDS + _AR_FIELDS

    def eval_sequence(self, data: SequenceData) -> Scores:
        # No special-cased "both empty" branch: with num_gt == num_dt == 0 every
        # per-label array below is simply empty, which concatenates in
        # `combine_sequences` as a no-op -- numerically identical to upstream's
        # `res[idx] = None` exclusion, without needing a non-array sentinel value.
        num_gt = data.num_gt_ids
        num_dt = data.num_pred_ids
        gt_frames, gt_lengths, gt_areas, _ = _group_tracks(data.gt_ids, data.gt_boxes, num_gt)
        dt_frames, dt_lengths, dt_areas, dt_scores = _group_tracks(
            data.pred_ids, data.pred_boxes, num_dt, data.pred_confidences
        )
        dt_scores_arr = np.asarray(dt_scores)

        gt_ig_masks = _ignore_masks(num_gt, gt_lengths, gt_areas)
        dt_ig_masks = _ignore_masks(num_dt, dt_lengths, dt_areas)

        ious = np.zeros((num_dt, num_gt))
        for i in range(num_dt):
            for j in range(num_gt):
                ious[i, j] = _track_iou(dt_frames[i], gt_frames[j])

        num_thrs = len(IOU_THRESHOLDS)
        scores: Scores = {_DT_SCORES: dt_scores_arr}
        for mask_idx, lbl in enumerate(LABELS):
            gt_ig_mask = gt_ig_masks[mask_idx]
            gt_order = np.argsort(gt_ig_mask, kind="mergesort")
            ious_sorted = ious[:, gt_order]
            gt_ig = gt_ig_mask[gt_order]

            gt_taken = np.zeros((num_thrs, num_gt), dtype=bool)
            dt_m = np.zeros((num_thrs, num_dt)) - 1
            dt_ig = np.zeros((num_thrs, num_dt))

            for thr_idx, iou_thr in enumerate(IOU_THRESHOLDS):
                if num_dt == 0:
                    break
                for dt_idx in range(num_dt):
                    iou = min(iou_thr, 1 - 1e-10)
                    m = -1
                    for gi in range(num_gt):
                        if gt_taken[thr_idx, gi]:
                            continue
                        if m > -1 and gt_ig[m] == 0 and gt_ig[gi] == 1:
                            break
                        if ious_sorted[dt_idx, gi] < iou - EPS:
                            continue
                        iou = ious_sorted[dt_idx, gi]
                        m = gi
                    if m == -1:
                        continue
                    dt_ig[thr_idx, dt_idx] = gt_ig[m]
                    dt_m[thr_idx, dt_idx] = m
                    gt_taken[thr_idx, m] = True

            dt_ig_mask = np.repeat(dt_ig_masks[mask_idx].reshape(1, num_dt), num_thrs, axis=0)
            dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == -1, dt_ig_mask))

            scores[f"{_DT_MATCHES}{lbl}"] = dt_m
            scores[f"{_DT_IGNORE}{lbl}"] = dt_ig
            scores[f"{_GT_IGNORE}{lbl}"] = gt_ig

        return scores

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        num_thrs = len(IOU_THRESHOLDS)
        num_recalls = len(RECALL_THRESHOLDS)
        precision = -np.ones((num_thrs, num_recalls, len(LABELS)))
        recall = -np.ones((num_thrs, len(LABELS)))
        det_weight = np.zeros((num_thrs, len(LABELS)))

        seq_results = list(all_res.values())
        for lbl_idx, lbl in enumerate(LABELS):
            dt_scores = np.concatenate([res[_DT_SCORES] for res in seq_results], axis=0)
            dt_order = np.argsort(-dt_scores, kind="mergesort")
            dt_m = np.concatenate([res[f"{_DT_MATCHES}{lbl}"] for res in seq_results], axis=1)[
                :, dt_order
            ]
            dt_ig = np.concatenate([res[f"{_DT_IGNORE}{lbl}"] for res in seq_results], axis=1)[
                :, dt_order
            ]
            gt_ig = np.concatenate([res[f"{_GT_IGNORE}{lbl}"] for res in seq_results], axis=0)
            num_gt = int(np.count_nonzero(gt_ig == 0))
            if num_gt == 0:
                continue

            tps = np.logical_and(dt_m != -1, np.logical_not(dt_ig))
            fps = np.logical_and(dt_m == -1, np.logical_not(dt_ig))
            tp_sum = np.cumsum(tps, axis=1).astype(float)
            fp_sum = np.cumsum(fps, axis=1).astype(float)

            for thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum, strict=True)):
                num_tp = len(tp)
                rc = tp / num_gt
                recall[thr_idx, lbl_idx] = rc[-1] if num_tp else 0.0

                pr = (tp / (fp + tp + np.spacing(1))).tolist()
                for i in range(num_tp - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                rec_thrs_insert_idx = np.searchsorted(rc, RECALL_THRESHOLDS, side="left")
                pr_at_recall = [0.0] * num_recalls
                try:
                    for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                        pr_at_recall[_idx] = pr[pr_idx]
                except IndexError:
                    pass
                precision[thr_idx, :, lbl_idx] = np.array(pr_at_recall)
                det_weight[thr_idx, lbl_idx] = np.count_nonzero(~dt_ig[thr_idx])

        scores: Scores = {}
        for lbl_idx, lbl in enumerate(LABELS):
            ap = np.zeros(num_thrs)
            for a in range(num_thrs):
                p = precision[a, :, lbl_idx]
                valid = p[p > -1]
                ap[a] = float(np.mean(valid)) if len(valid) else -1.0
            scores[f"AP_{lbl}"] = ap
            scores[f"AR_{lbl}"] = recall[:, lbl_idx].copy()
            scores[_weight_field(lbl)] = det_weight[:, lbl_idx]
        return scores

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        # Upstream's `ignore_empty_classes` parameter is deliberately unsupported here;
        # this always averages over every class in `all_res`.
        scores: Scores = {}
        for field in self.fields:
            stacked = np.array([res[field] for res in all_res.values()])
            out = np.zeros(len(IOU_THRESHOLDS))
            for a in range(len(IOU_THRESHOLDS)):
                values = stacked[:, a]
                valid = values[values > -1]
                out[a] = float(np.mean(valid)) if len(valid) else -1.0
            scores[field] = out
        return scores

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        # DIVERGENCE (the sole one, see README's "Known divergences from TrackEval"):
        # upstream's `combine_classes_det_averaged` is byte-for-byte the class-averaged
        # combiner -- it never actually weights by detections. moteval instead weights
        # each class's contribution by `_num_dt_<lbl>`, the count of non-ignored
        # detections that fed that class's precision/recall curve for that label and
        # IoU threshold (populated by `combine_sequences`), falling back to an
        # unweighted mean when every valid class has zero weight.
        scores: Scores = {}
        for lbl in LABELS:
            weights = np.array([res[_weight_field(lbl)] for res in all_res.values()])
            for field in (f"AP_{lbl}", f"AR_{lbl}"):
                values = np.array([res[field] for res in all_res.values()])
                out = np.zeros(len(IOU_THRESHOLDS))
                for a in range(len(IOU_THRESHOLDS)):
                    col = values[:, a]
                    valid = col > -1
                    if not np.any(valid):
                        out[a] = -1.0
                    elif weights[valid, a].sum() > 0:
                        out[a] = float(np.average(col[valid], weights=weights[valid, a]))
                    else:
                        out[a] = float(np.mean(col[valid]))
                scores[field] = out
        return scores
