"""J&F metric: DAVIS-style region similarity J and boundary accuracy F on masks.

TrackEval's `JAndF` exactly (per-track mean/recall/decay + the J&F average), including
its quirks: zero-mask padding for timesteps where a track has no detection (and whole
padded tracker tracks when there are fewer tracker ids than GT ids), the
both-masks-empty IoU := 1 rule, the decay bin edges cast to uint8 (which wraps for
sequences longer than 255 frames), and GT-to-pred assignment on mean-over-time J only.
Boundary matches are counted exactly on integer boundary coordinates, as upstream's
disk dilation counts them. Implemented in ``src/metrics/jf.rs``.
"""

from moteval.metrics.base import RustMetric, rust_metric_info


class JAndF(RustMetric):
    rust_name = "JAndF"
    fields, geometry = rust_metric_info("JAndF")
