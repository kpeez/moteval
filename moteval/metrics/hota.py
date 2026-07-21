"""HOTA metric family: HOTA/DetA/AssA/DetRe/DetPr/AssRe/AssPr/LocA/OWTA over
19 alpha thresholds (0.05..0.95), plus HOTA(0)/LocA(0)/HOTALocA(0).

Ports TrackEval's global alignment (Hungarian assignment weighted by a
Jaccard-normalized potential-match count) and eps-guarded alpha thresholding
exactly, so per-field results are bit-identical to the oracle.
"""

import numpy as np

from moteval.data.model import SequenceData
from moteval.metrics._matching import EPS, linear_sum_assignment
from moteval.metrics.base import Metric, Scores

ALPHAS = np.arange(0.05, 0.99, 0.05)

_FLOAT_ARRAY_FIELDS = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA")
_INTEGER_ARRAY_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")
_FLOAT_FIELDS = ("HOTA(0)", "LocA(0)", "HOTALocA(0)")

Arrays = dict[str, np.ndarray]


class HOTA(Metric):
    fields = _FLOAT_ARRAY_FIELDS + _INTEGER_ARRAY_FIELDS + _FLOAT_FIELDS

    def eval_sequence(self, data: SequenceData) -> Scores:
        num_alphas = len(ALPHAS)
        arrays: Arrays = {
            f: np.zeros(num_alphas) for f in _FLOAT_ARRAY_FIELDS + _INTEGER_ARRAY_FIELDS
        }

        if data.num_pred_dets == 0:
            arrays["HOTA_FN"] = np.full(num_alphas, data.num_gt_dets, dtype=float)
            arrays["LocA"] = np.ones(num_alphas)
            return self._finalize(arrays)
        if data.num_gt_dets == 0:
            arrays["HOTA_FP"] = np.full(num_alphas, data.num_pred_dets, dtype=float)
            arrays["LocA"] = np.ones(num_alphas)
            return self._finalize(arrays)

        potential_matches_count = np.zeros((data.num_gt_ids, data.num_pred_ids))
        gt_id_count = np.zeros((data.num_gt_ids, 1))
        pred_id_count = np.zeros((1, data.num_pred_ids))

        for gt_ids_t, pred_ids_t, similarity in zip(
            data.gt_ids, data.pred_ids, data.similarity, strict=True
        ):
            if len(gt_ids_t) > 0 and len(pred_ids_t) > 0:
                sim_iou_denom = (
                    similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
                )
                sim_iou = np.zeros_like(similarity)
                sim_iou_mask = sim_iou_denom > 0 + EPS
                sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
                potential_matches_count[gt_ids_t[:, np.newaxis], pred_ids_t[np.newaxis, :]] += (
                    sim_iou
                )
            gt_id_count[gt_ids_t] += 1
            pred_id_count[0, pred_ids_t] += 1

        global_alignment_score = potential_matches_count / (
            gt_id_count + pred_id_count - potential_matches_count
        )

        num_gt_ids, num_pred_ids = data.num_gt_ids, data.num_pred_ids
        alpha_thresholds = ALPHAS - EPS
        match_flat_indices: list[np.ndarray] = []

        for gt_ids_t, pred_ids_t, similarity in zip(
            data.gt_ids, data.pred_ids, data.similarity, strict=True
        ):
            if len(gt_ids_t) == 0:
                arrays["HOTA_FP"] += len(pred_ids_t)
                continue
            if len(pred_ids_t) == 0:
                arrays["HOTA_FN"] += len(gt_ids_t)
                continue

            score_mat = (
                global_alignment_score[gt_ids_t[:, np.newaxis], pred_ids_t[np.newaxis, :]]
                * similarity
            )
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Vectorized across the 19 alphas; every quantity below is either an
            # order-independent integer count or (LocA) a strict left-to-right
            # float scan identical to upstream's accumulation.
            matched_sims = similarity[match_rows, match_cols]
            matched = matched_sims[:, np.newaxis] >= alpha_thresholds[np.newaxis, :]
            num_matches = matched.sum(0)
            arrays["HOTA_TP"] += num_matches
            arrays["HOTA_FN"] += len(gt_ids_t) - num_matches
            arrays["HOTA_FP"] += len(pred_ids_t) - num_matches
            if len(matched_sims) > 0:
                # np.add.accumulate is a single-accumulator left-to-right scan
                # (upstream uses builtin sum, never ndarray.sum's 8-way pairwise
                # unroll); masked-out entries add exactly 0.0, an exact no-op.
                arrays["LocA"] += np.add.accumulate(
                    np.where(matched, matched_sims[:, np.newaxis], 0.0), axis=0
                )[-1]
                pair_flat = gt_ids_t[match_rows] * num_pred_ids + pred_ids_t[match_cols]
                pair_idx, alpha_idx = np.nonzero(matched)
                match_flat_indices.append(
                    alpha_idx * (num_gt_ids * num_pred_ids) + pair_flat[pair_idx]
                )

        matches_counts = np.bincount(
            np.concatenate(match_flat_indices)
            if match_flat_indices
            else np.empty(0, dtype=np.int64),
            minlength=len(ALPHAS) * num_gt_ids * num_pred_ids,
        ).reshape(len(ALPHAS), num_gt_ids, num_pred_ids)

        for a, _alpha in enumerate(ALPHAS):
            matches_count = matches_counts[a].astype(float)
            ass_a = matches_count / np.maximum(1, gt_id_count + pred_id_count - matches_count)
            arrays["AssA"][a] = np.sum(matches_count * ass_a) / np.maximum(1, arrays["HOTA_TP"][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            arrays["AssRe"][a] = np.sum(matches_count * ass_re) / np.maximum(
                1, arrays["HOTA_TP"][a]
            )
            ass_pr = matches_count / np.maximum(1, pred_id_count)
            arrays["AssPr"][a] = np.sum(matches_count * ass_pr) / np.maximum(
                1, arrays["HOTA_TP"][a]
            )

        arrays["LocA"] = np.maximum(1e-10, arrays["LocA"]) / np.maximum(1e-10, arrays["HOTA_TP"])
        arrays = self._compute_det_fields(arrays)
        return self._finalize(arrays)

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        arrays: Arrays = {}
        for field in _INTEGER_ARRAY_FIELDS:
            arrays[field] = np.asarray(self._combine_sum(all_res, field))
        for field in ("AssRe", "AssPr", "AssA"):
            arrays[field] = np.asarray(
                self._combine_weighted_av(all_res, field, arrays, weight_field="HOTA_TP")
            )
        loca_weighted_sum = sum(scores["LocA"] * scores["HOTA_TP"] for scores in all_res.values())
        arrays["LocA"] = np.asarray(
            np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, arrays["HOTA_TP"])
        )
        arrays = self._compute_det_fields(arrays)
        return self._finalize(arrays)

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        # Upstream's `ignore_empty_classes` parameter is deliberately unsupported here;
        # this always averages over every class in `all_res`.
        arrays: Arrays = {}
        for field in _INTEGER_ARRAY_FIELDS:
            arrays[field] = np.asarray(self._combine_sum(all_res, field))
        for field in _FLOAT_ARRAY_FIELDS:
            arrays[field] = np.mean([scores[field] for scores in all_res.values()], axis=0)
        scores_out: Scores = dict(arrays)
        for field in _FLOAT_FIELDS:
            scores_out[field] = float(
                np.mean([scores[field] for scores in all_res.values()], axis=0)
            )
        return scores_out

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return self.combine_sequences(all_res)

    @staticmethod
    def _compute_det_fields(arrays: Arrays) -> Arrays:
        arrays["DetRe"] = arrays["HOTA_TP"] / np.maximum(1, arrays["HOTA_TP"] + arrays["HOTA_FN"])
        arrays["DetPr"] = arrays["HOTA_TP"] / np.maximum(1, arrays["HOTA_TP"] + arrays["HOTA_FP"])
        arrays["DetA"] = arrays["HOTA_TP"] / np.maximum(
            1, arrays["HOTA_TP"] + arrays["HOTA_FN"] + arrays["HOTA_FP"]
        )
        arrays["HOTA"] = np.sqrt(arrays["DetA"] * arrays["AssA"])
        arrays["OWTA"] = np.sqrt(arrays["DetRe"] * arrays["AssA"])
        return arrays

    @staticmethod
    def _finalize(arrays: Arrays) -> Scores:
        hota_0 = float(arrays["HOTA"][0])
        loca_0 = float(arrays["LocA"][0])
        scores: Scores = dict(arrays)
        scores["HOTA(0)"] = hota_0
        scores["LocA(0)"] = loca_0
        scores["HOTALocA(0)"] = hota_0 * loca_0
        return scores
