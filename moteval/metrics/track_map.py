"""TrackMAP metric: track-level average precision/recall over IoU thresholds,
computed separately over the whole track set and over area/time-length subsets.

TrackEval's TAO-style algorithm: per-track box IoU (summed intersection/union over the
union of frames either track appears in, in CPython ``set`` iteration order), greedy
per-IoU-threshold matching with gt sorted ignore-last, right-to-left monotonic precision
interpolation, and ``np.searchsorted``-based recall-threshold indexing, bit-identical to
the oracle. Matched GT state is an explicit boolean array (upstream compares against the
matched track id, a latent bug for id 0, which densified ids use).

Documented divergence (the sole permitted one, see README's "Known divergences"):
upstream's ``combine_classes_det_averaged`` is a copy-paste of the class-averaged
combiner (never actually weights by detections). moteval implements the intended
behavior instead. Implemented in ``rust/moteval/src/metrics/track_map.rs``.
"""

from moteval.metrics.base import RustMetric, rust_metric_info


class TrackMAP(RustMetric):
    rust_name = "TrackMAP"
    fields, geometry = rust_metric_info("TrackMAP")
