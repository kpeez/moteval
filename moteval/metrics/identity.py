"""Identity (ID) metric family: IDF1/IDR/IDP/IDTP/IDFN/IDFP.

TrackEval's global track-level matching on the
(num_gt_ids + num_pred_ids) x (num_gt_ids + num_pred_ids) block cost matrix, whose
off-diagonal blocks are filled with 1e10, and its non-eps-guarded similarity threshold
(``similarity >= threshold``), bit-identical to the oracle. Implemented in
``src/metrics/identity.rs``.
"""

from moteval.metrics.base import RustMetric, rust_metric_info


class Identity(RustMetric):
    rust_name = "Identity"
    fields, geometry = rust_metric_info("Identity")
