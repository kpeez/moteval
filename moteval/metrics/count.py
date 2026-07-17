"""Count metric: detection and identity totals (no matching involved)."""

from moteval.data.model import SequenceData
from moteval.metrics.base import Metric, Scores


class Count(Metric):
    fields = ("Dets", "GT_Dets", "IDs", "GT_IDs")

    def eval_sequence(self, data: SequenceData) -> Scores:
        return {
            "Dets": float(data.num_pred_dets),
            "GT_Dets": float(data.num_gt_dets),
            "IDs": float(data.num_pred_ids),
            "GT_IDs": float(data.num_gt_ids),
        }

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        return {f: self._combine_sum(all_res, f) for f in self.fields}

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return {f: self._combine_sum(all_res, f) for f in self.fields}

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return {f: self._combine_sum(all_res, f) for f in self.fields}
