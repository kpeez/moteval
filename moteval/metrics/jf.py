"""J&F metric: DAVIS-style region similarity J and boundary accuracy F on masks.

Ports TrackEval's `JAndF` exactly (per-track mean/recall/decay + the J&F
average), including its quirks: zero-mask padding encoded from a Fortran-order
array for timesteps where a track has no detection (and whole padded tracker
tracks when there are fewer tracker ids than GT ids), the both-masks-empty
IoU := 1 rule, the decay bin edges cast to uint8 (which wraps for sequences
longer than 255 frames — upstream behavior, replicated), and GT-to-pred
assignment on mean-over-time J only (upstream's default ``optim_type='J'``;
its unreachable 'J&F' branch is not ported). Unlike other metrics, J&F reads
mask geometry directly rather than precomputed similarity.

Upstream dilates boundary maps with ``cv2.dilate`` + ``skimage.morphology.disk``
and intersects each boundary with the other's dilation; this port counts the
identical matches via nearest-neighbor queries on the sparse boundary
coordinates (a pixel is inside the disk dilation iff its nearest boundary pixel
is within the disk radius), which is exact for integer coordinates and avoids
both the heavy dependencies and full-frame dilation cost (seconds per frame at
1080p). Verified against the oracle's real cv2/skimage path by the parity suite.
"""

import numpy as np
from pycocotools import mask as mask_utils
from scipy.spatial import KDTree

from moteval.data.model import SequenceData
from moteval.metrics._matching import linear_sum_assignment
from moteval.metrics.base import Metric, Scores

_EPS = float(np.finfo("float").eps)
_BOUND_TH = 0.008
_N_BINS = 4
_FLOAT_FIELDS = ("J-Mean", "J-Recall", "J-Decay", "F-Mean", "F-Recall", "F-Decay", "J&F")


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Binary boundary map with 1-pixel-wide boundaries offset half a pixel
    towards the origin (David Martin, 2003), as vendored by TrackEval. The
    unused rescale branch (bmap smaller than seg) is not ported."""
    seg = seg.astype(bool)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def _boundary_points(rle: dict) -> np.ndarray:
    """`np.argwhere(_seg2bmap(decode(rle)))`, computed on the mask's bbox crop.

    `_seg2bmap` boundaries sit on mask pixels or one pixel to their top/left,
    so a 1px top/left margin captures them all; the clamped 1px bottom/right
    margin makes the crop's last row/col coincide with the frame's whenever the
    bbox touches the frame edge, so `_seg2bmap`'s edge fixups are unchanged.
    Rows/cols outside the crop can never hold boundary pixels (all-False
    neighborhoods). Point sets verified identical to the full-frame path over
    every mask of the MOTS20 parity workload, including frame-edge masks.
    """
    height, width = rle["size"]
    x, y, bw, bh = mask_utils.toBbox(rle).astype(int)
    r0, c0 = max(0, y - 1), max(0, x - 1)
    r1, c1 = min(height, y + bh + 1), min(width, x + bw + 1)
    boundary = _seg2bmap(mask_utils.decode(rle)[r0:r1, c0:c1])
    return np.argwhere(boundary) + (r0, c0)


def _compute_f(
    gt_dets: list,
    pred_dets: list,
    pred_id: int,
    gt_id: int,
    gt_areas: list[np.ndarray],
    pred_areas: list[np.ndarray],
) -> np.ndarray:
    f = np.zeros(len(gt_dets))
    for t, (gt_masks, pred_masks) in enumerate(zip(gt_dets, pred_dets, strict=True)):
        # A mask's `_seg2bmap` boundary is empty iff its area is 0 (no pixels)
        # or H*W (full frame: every transition-free row/col plus the edge
        # fixups yields no boundary). Both cases are decided from the RLE area
        # without decoding, and produce the same precision/recall the original
        # empty-boundary branches assigned.
        height, width = pred_masks[pred_id]["size"]
        full_area = height * width
        fg_nonempty = 0 < pred_areas[t][pred_id] < full_area
        gt_nonempty = 0 < gt_areas[t][gt_id] < full_area
        if not fg_nonempty and not gt_nonempty:
            f[t] = 1.0  # precision = recall = 1
            continue
        if not fg_nonempty or not gt_nonempty:
            f[t] = 0.0  # one of precision/recall is 0, the other 1
            continue

        bound_pix = (
            _BOUND_TH
            if _BOUND_TH >= 1 - _EPS
            else np.ceil(_BOUND_TH * np.linalg.norm((height, width)))
        )
        # Upstream dilates each boundary by an L2 disk of radius r=int(bound_pix)
        # and counts the other boundary's pixels inside the dilation. A pixel is
        # inside the dilation iff its nearest boundary pixel lies within distance
        # r, so nearest-neighbor queries on the sparse boundary coordinates give
        # the identical counts without materializing full-frame dilations (which
        # cost seconds per frame at 1080p). Exact despite float distances: for
        # integer coordinates and integer r, d <= r never misrounds because
        # correctly-rounded sqrt preserves ordering against the representable r.
        r = int(bound_pix)
        fg_pts = _boundary_points(pred_masks[pred_id])
        gt_pts = _boundary_points(gt_masks[gt_id])
        fg_match = np.count_nonzero(KDTree(gt_pts).query(fg_pts, k=1)[0] <= r)
        gt_match = np.count_nonzero(KDTree(fg_pts).query(gt_pts, k=1)[0] <= r)
        precision = fg_match / float(len(fg_pts))
        recall = gt_match / float(len(gt_pts))

        f[t] = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f


def _compute_j(
    gt_dets: list, pred_dets: list, num_gt_ids: int, num_pred_ids: int, num_timesteps: int
) -> np.ndarray:
    j = np.zeros((num_pred_ids, num_gt_ids, num_timesteps))
    for t, (time_gt, time_pred) in enumerate(zip(gt_dets, pred_dets, strict=True)):
        area_gt = mask_utils.area(time_gt)
        time_pred = list(time_pred)
        area_pred = mask_utils.area(time_pred)

        area_pred = np.repeat(area_pred[:, np.newaxis], len(area_gt), axis=1)
        area_gt = np.repeat(area_gt[np.newaxis, :], len(area_pred), axis=0)

        ious = np.atleast_2d(mask_utils.iou(time_pred, time_gt, [0] * len(time_gt)))
        # both masks (near-)empty in this timestep counts as a perfect match
        ious[np.isclose(area_pred, 0) & np.isclose(area_gt, 0)] = 1
        j[..., t] = ious
    return j


class JAndF(Metric):
    fields = _FLOAT_FIELDS + ("num_gt_tracks",)

    def eval_sequence(self, data: SequenceData) -> Scores:
        num_timesteps = data.num_timesteps
        num_gt_ids = data.num_gt_ids
        num_pred_ids = data.num_pred_ids

        frame_shape = None
        if num_gt_ids > 0:
            for t in range(num_timesteps):
                if len(data.gt_ids[t]) > 0:
                    frame_shape = data.gt_masks[t][0]["size"]
                    break
        elif num_pred_ids > 0:
            for t in range(num_timesteps):
                if len(data.pred_ids[t]) > 0:
                    frame_shape = data.pred_masks[t][0]["size"]
                    break

        gt_dets = [list(data.gt_masks[t]) for t in range(num_timesteps)]
        pred_dets = [list(data.pred_masks[t]) for t in range(num_timesteps)]
        if frame_shape:
            # pad timesteps where a track has no detection with an all-zero mask,
            # encoded from a Fortran-order array exactly as upstream does
            padding_mask = mask_utils.encode(np.zeros(frame_shape, order="F").astype(np.uint8))
            for t in range(num_timesteps):
                gt_mapping = dict(zip(data.gt_ids[t].tolist(), gt_dets[t], strict=True))
                gt_dets[t] = [gt_mapping.get(index, padding_mask) for index in range(num_gt_ids)]
                pred_mapping = dict(zip(data.pred_ids[t].tolist(), pred_dets[t], strict=True))
                pred_dets[t] = [
                    pred_mapping.get(index, padding_mask) for index in range(num_pred_ids)
                ]
            if num_pred_ids < num_gt_ids:
                diff = num_gt_ids - num_pred_ids
                for t in range(num_timesteps):
                    pred_dets[t] = pred_dets[t] + [padding_mask for _ in range(diff)]
                num_pred_ids += diff

        j = _compute_j(gt_dets, pred_dets, num_gt_ids, num_pred_ids, num_timesteps)

        # RLE-side per-frame areas for _compute_f's empty-boundary fast path
        gt_areas = [mask_utils.area(dets) for dets in gt_dets]
        pred_areas = [mask_utils.area(dets) for dets in pred_dets]

        # assignment on mean-over-time J (upstream default optim_type='J');
        # F is computed only for assigned pairs
        optim_metrics = np.mean(j, axis=2)
        row_ind, col_ind = linear_sum_assignment(-optim_metrics)
        j_m = j[row_ind, col_ind, :]
        f_m = np.zeros_like(j_m)
        for i, (pred_ind, gt_ind) in enumerate(zip(row_ind, col_ind, strict=True)):
            f_m[i] = _compute_f(gt_dets, pred_dets, pred_ind, gt_ind, gt_areas, pred_areas)

        # append zero rows for unmatched GT tracks (false negatives)
        if j_m.shape[0] < data.num_gt_ids:
            diff = data.num_gt_ids - j_m.shape[0]
            j_m = np.concatenate((j_m, np.zeros((diff, j_m.shape[1]))), axis=0)
            f_m = np.concatenate((f_m, np.zeros((diff, f_m.shape[1]))), axis=0)

        res: dict[str, list | float | np.ndarray] = {
            "J-Mean": [np.nanmean(j_m[i, :]) for i in range(j_m.shape[0])],
            "J-Recall": [np.nanmean(j_m[i, :] > 0.5 + _EPS) for i in range(j_m.shape[0])],
            "F-Mean": [np.nanmean(f_m[i, :]) for i in range(f_m.shape[0])],
            "F-Recall": [np.nanmean(f_m[i, :] > 0.5 + _EPS) for i in range(f_m.shape[0])],
            "J-Decay": [],
            "F-Decay": [],
        }
        # uint8 cast replicated from upstream: bin edges wrap for sequences
        # longer than 255 frames
        ids = np.round(np.linspace(1, num_timesteps, _N_BINS + 1) + 1e-10) - 1
        ids = ids.astype(np.uint8)
        for k in range(j_m.shape[0]):
            d_bins_j = [j_m[k][ids[i] : ids[i + 1] + 1] for i in range(_N_BINS)]
            res["J-Decay"].append(np.nanmean(d_bins_j[0]) - np.nanmean(d_bins_j[3]))
        for k in range(f_m.shape[0]):
            d_bins_f = [f_m[k][ids[i] : ids[i + 1] + 1] for i in range(_N_BINS)]
            res["F-Decay"].append(np.nanmean(d_bins_f[0]) - np.nanmean(d_bins_f[3]))

        scores: Scores = {"num_gt_tracks": float(len(res["J-Mean"]))}
        for field in _FLOAT_FIELDS[:-1]:
            scores[field] = float(np.mean(res[field]))
        scores["J&F"] = (scores["J-Mean"] + scores["F-Mean"]) / 2
        return scores

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        res: Scores = {"num_gt_tracks": self._combine_sum(all_res, "num_gt_tracks")}
        for field in _FLOAT_FIELDS:
            res[field] = self._combine_weighted_av(
                all_res, field, res, weight_field="num_gt_tracks"
            )
        return res

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        res: Scores = {"num_gt_tracks": self._combine_sum(all_res, "num_gt_tracks")}
        for field in _FLOAT_FIELDS:
            res[field] = float(np.mean([scores[field] for scores in all_res.values()]))
        return res

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        # Upstream's two class-combine modes coincide for J&F; keep them in lockstep.
        return self.combine_classes_class_averaged(all_res)
