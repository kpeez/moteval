from moteval.metrics.base import RustMetric, rust_metric_info


class TrackMAP(RustMetric):
    """``combine_classes_det_averaged`` weights classes by detection count; TrackEval
    ``12c8791b`` repeats its class average there.
    """

    rust_name = "TrackMAP"
    fields, geometry = rust_metric_info("TrackMAP")
