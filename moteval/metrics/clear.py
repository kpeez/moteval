"""CLEAR (MOTChallenge) metric family: MOTA/MOTP/MODA/sMOTA/CLR_Re/CLR_Pr/CLR_F1/
MT/PT/ML/MTR/PTR/MLR/Frag/IDSW/CLR_TP/CLR_FN/CLR_FP and friends.

Ports TrackEval's per-frame Hungarian matching, which applies a 1000x cost bonus
for continuing the previous frame's GT-track/pred-ID pairing before assignment,
and its eps-guarded similarity threshold (``similarity < threshold - eps``), so
per-field results are bit-identical to the oracle.
"""

import numpy as np

from moteval.data.model import SequenceData
from moteval.metrics._matching import EPS, linear_sum_assignment
from moteval.metrics.base import Metric, Scores

THRESHOLD = 0.5

_INTEGER_FIELDS = ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag", "CLR_Frames")
_FLOAT_FIELDS = (
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "sMOTA",
    "CLR_F1",
    "FP_per_frame",
    "MOTAL",
    "MOTP_sum",
)
_SUMMED_FIELDS = _INTEGER_FIELDS + ("MOTP_sum",)


class CLEAR(Metric):
    fields = _FLOAT_FIELDS + _INTEGER_FIELDS

    def eval_sequence(self, data: SequenceData) -> Scores:
        res: Scores = dict.fromkeys(self.fields, 0.0)

        if data.num_pred_dets == 0:
            res["CLR_FN"] = float(data.num_gt_dets)
            res["ML"] = float(data.num_gt_ids)
            res["MLR"] = 1.0
            return res
        if data.num_gt_dets == 0:
            res["CLR_FP"] = float(data.num_pred_dets)
            res["MLR"] = 1.0
            return res

        num_gt_ids = data.num_gt_ids
        gt_id_count = np.zeros(num_gt_ids)
        gt_matched_count = np.zeros(num_gt_ids)
        gt_frag_count = np.zeros(num_gt_ids)

        # IDSW is scored against the last frame each gt_id was matched (any number of
        # frames back), but only the immediately preceding timestep feeds the matching bonus.
        prev_tracker_id = np.nan * np.zeros(num_gt_ids)
        prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)

        for gt_ids_t, pred_ids_t, similarity in zip(
            data.gt_ids, data.pred_ids, data.similarity, strict=True
        ):
            if len(gt_ids_t) == 0:
                res["CLR_FP"] += len(pred_ids_t)
                continue
            if len(pred_ids_t) == 0:
                res["CLR_FN"] += len(gt_ids_t)
                gt_id_count[gt_ids_t] += 1
                continue

            score_mat = (
                pred_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
            )
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < THRESHOLD - EPS] = 0

            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = score_mat[match_rows, match_cols] > 0 + EPS
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_pred_ids = pred_ids_t[match_cols]

            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            is_idsw = (~np.isnan(prev_matched_tracker_ids)) & (
                matched_pred_ids != prev_matched_tracker_ids
            )
            res["IDSW"] += np.sum(is_idsw)

            gt_id_count[gt_ids_t] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)
            prev_tracker_id[matched_gt_ids] = matched_pred_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_pred_ids
            currently_tracked = ~np.isnan(prev_timestep_tracker_id)
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

            num_matches = len(matched_gt_ids)
            res["CLR_TP"] += num_matches
            res["CLR_FN"] += len(gt_ids_t) - num_matches
            res["CLR_FP"] += len(pred_ids_t) - num_matches
            if num_matches > 0:
                res["MOTP_sum"] += sum(similarity[match_rows, match_cols])

        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res["MT"] = float(np.sum(tracked_ratio > 0.8))
        res["PT"] = float(np.sum(tracked_ratio >= 0.2)) - res["MT"]
        res["ML"] = num_gt_ids - res["MT"] - res["PT"]
        res["Frag"] = float(np.sum(gt_frag_count[gt_frag_count > 0] - 1))
        res["CLR_Frames"] = float(data.num_timesteps)

        # Upstream also assigns MOTP here before _compute_final_fields recomputes it;
        # that first assignment is redundant, so it is deliberately omitted.
        return self._compute_final_fields(res)

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        res: Scores = {field: self._combine_sum(all_res, field) for field in _SUMMED_FIELDS}
        return self._compute_final_fields(res)

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return self.combine_sequences(all_res)

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        # Upstream's `ignore_empty_classes` parameter is deliberately unsupported here
        # (safe: upstream's evaluator only ever calls this with its default False);
        # this always averages over every class in `all_res`. Note float fields (which
        # includes MOTP_sum) are averaged here, not recomputed from summed counts.
        res: Scores = {}
        for field in _INTEGER_FIELDS:
            res[field] = self._combine_sum(all_res, field)
        for field in _FLOAT_FIELDS:
            res[field] = float(np.mean([scores[field] for scores in all_res.values()]))
        return res

    @staticmethod
    def _compute_final_fields(res: Scores) -> Scores:
        num_gt_ids = res["MT"] + res["ML"] + res["PT"]
        res["MTR"] = res["MT"] / max(1.0, num_gt_ids)
        res["MLR"] = res["ML"] / max(1.0, num_gt_ids)
        res["PTR"] = res["PT"] / max(1.0, num_gt_ids)
        res["CLR_Re"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + res["CLR_FN"])
        res["CLR_Pr"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + res["CLR_FP"])
        res["MODA"] = (res["CLR_TP"] - res["CLR_FP"]) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
        res["MOTA"] = (res["CLR_TP"] - res["CLR_FP"] - res["IDSW"]) / max(
            1.0, res["CLR_TP"] + res["CLR_FN"]
        )
        res["MOTP"] = res["MOTP_sum"] / max(1.0, res["CLR_TP"])
        res["sMOTA"] = (res["MOTP_sum"] - res["CLR_FP"] - res["IDSW"]) / max(
            1.0, res["CLR_TP"] + res["CLR_FN"]
        )
        res["CLR_F1"] = res["CLR_TP"] / max(
            1.0, res["CLR_TP"] + 0.5 * res["CLR_FN"] + 0.5 * res["CLR_FP"]
        )
        res["FP_per_frame"] = res["CLR_FP"] / max(1.0, res["CLR_Frames"])
        safe_log_idsw = np.log10(res["IDSW"]) if res["IDSW"] > 0 else res["IDSW"]
        res["MOTAL"] = (res["CLR_TP"] - res["CLR_FP"] - safe_log_idsw) / max(
            1.0, res["CLR_TP"] + res["CLR_FN"]
        )
        return res
