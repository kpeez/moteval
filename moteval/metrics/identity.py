from moteval.metrics.base import RustMetric, rust_metric_info


class Identity(RustMetric):
    rust_name = "Identity"
    fields, geometry = rust_metric_info("Identity")
