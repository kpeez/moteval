"""Identity (ID) metric family: IDF1/IDR/IDP/IDTP/IDFN/IDFP.

Ports TrackEval's global track-level matching on the
(num_gt_ids + num_pred_ids) x (num_gt_ids + num_pred_ids) block cost matrix, whose
off-diagonal blocks are filled with 1e10, solved with scipy's Hungarian solver, and
its non-eps-guarded similarity threshold (``similarity >= threshold``), so per-field
results are bit-identical to the oracle.
"""

import numpy as np

from moteval.data.model import SequenceData
from moteval.metrics._matching import linear_sum_assignment
from moteval.metrics.base import Metric, Scores

THRESHOLD = 0.5

_INTEGER_FIELDS = ("IDTP", "IDFN", "IDFP")
_FLOAT_FIELDS = ("IDF1", "IDR", "IDP")


class Identity(Metric):
    fields = _FLOAT_FIELDS + _INTEGER_FIELDS

    def eval_sequence(self, data: SequenceData) -> Scores:
        res: Scores = dict.fromkeys(self.fields, 0.0)

        if data.num_pred_dets == 0:
            res["IDFN"] = float(data.num_gt_dets)
            return res
        if data.num_gt_dets == 0:
            res["IDFP"] = float(data.num_pred_dets)
            return res

        num_gt_ids = data.num_gt_ids
        num_pred_ids = data.num_pred_ids
        potential_matches_count = np.zeros((num_gt_ids, num_pred_ids))
        gt_id_count = np.zeros(num_gt_ids)
        pred_id_count = np.zeros(num_pred_ids)

        for gt_ids_t, pred_ids_t, similarity in zip(
            data.gt_ids, data.pred_ids, data.similarity, strict=True
        ):
            matches_mask = similarity >= THRESHOLD
            match_idx_gt, match_idx_pred = np.nonzero(matches_mask)
            potential_matches_count[gt_ids_t[match_idx_gt], pred_ids_t[match_idx_pred]] += 1
            gt_id_count[gt_ids_t] += 1
            pred_id_count[pred_ids_t] += 1

        fp_mat = np.zeros((num_gt_ids + num_pred_ids, num_gt_ids + num_pred_ids))
        fn_mat = np.zeros((num_gt_ids + num_pred_ids, num_gt_ids + num_pred_ids))
        fp_mat[num_gt_ids:, :num_pred_ids] = 1e10
        fn_mat[:num_gt_ids, num_pred_ids:] = 1e10
        for gt_id in range(num_gt_ids):
            fn_mat[gt_id, :num_pred_ids] = gt_id_count[gt_id]
            fn_mat[gt_id, num_pred_ids + gt_id] = gt_id_count[gt_id]
        for pred_id in range(num_pred_ids):
            fp_mat[:num_gt_ids, pred_id] = pred_id_count[pred_id]
            fp_mat[pred_id + num_gt_ids, pred_id] = pred_id_count[pred_id]
        fn_mat[:num_gt_ids, :num_pred_ids] -= potential_matches_count
        fp_mat[:num_gt_ids, :num_pred_ids] -= potential_matches_count

        match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)
        res["IDFN"] = float(fn_mat[match_rows, match_cols].sum())
        res["IDFP"] = float(fp_mat[match_rows, match_cols].sum())
        res["IDTP"] = float(gt_id_count.sum() - res["IDFN"])

        return self._compute_final_fields(res)

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        res: Scores = {field: self._combine_sum(all_res, field) for field in _INTEGER_FIELDS}
        return self._compute_final_fields(res)

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return self.combine_sequences(all_res)

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        # Upstream's `ignore_empty_classes` parameter is deliberately unsupported here;
        # this always averages over every class in `all_res`. Float fields are averaged
        # directly here, not recomputed from summed counts (same quirk as CLEAR).
        res: Scores = {field: self._combine_sum(all_res, field) for field in _INTEGER_FIELDS}
        for field in _FLOAT_FIELDS:
            res[field] = float(np.mean([scores[field] for scores in all_res.values()]))
        return res

    @staticmethod
    def _compute_final_fields(res: Scores) -> Scores:
        res["IDR"] = res["IDTP"] / max(1.0, res["IDTP"] + res["IDFN"])
        res["IDP"] = res["IDTP"] / max(1.0, res["IDTP"] + res["IDFP"])
        res["IDF1"] = res["IDTP"] / max(1.0, res["IDTP"] + 0.5 * res["IDFP"] + 0.5 * res["IDFN"])
        return res
