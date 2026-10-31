"""CLEAR (MOTChallenge) metric family: MOTA/MOTP/MODA/sMOTA/CLR_Re/CLR_Pr/CLR_F1/
MT/PT/ML/MTR/PTR/MLR/Frag/IDSW/CLR_TP/CLR_FN/CLR_FP and friends.

TrackEval's per-frame Hungarian matching, with its 1000x cost bonus for continuing the
previous frame's GT-track/pred-ID pairing and its eps-guarded similarity threshold
(``similarity < threshold - eps``), bit-identical to the oracle. Implemented in
``rust/moteval/src/metrics/clear.rs``.
"""

from moteval.metrics.base import RustMetric, rust_metric_info


class CLEAR(RustMetric):
    rust_name = "CLEAR"
    fields, geometry = rust_metric_info("CLEAR")
