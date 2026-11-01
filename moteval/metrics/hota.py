"""HOTA metric family: HOTA/DetA/AssA/DetRe/DetPr/AssRe/AssPr/LocA/OWTA over
19 alpha thresholds (0.05..0.95), plus HOTA(0)/LocA(0)/HOTALocA(0).

TrackEval's global alignment (Hungarian assignment weighted by a Jaccard-normalized
potential-match count) and eps-guarded alpha thresholding, bit-identical to the oracle.
Implemented in ``src/metrics/hota.rs``.
"""

from moteval.metrics.base import RustMetric, rust_metric_info


class HOTA(RustMetric):
    rust_name = "HOTA"
    fields, geometry = rust_metric_info("HOTA")
