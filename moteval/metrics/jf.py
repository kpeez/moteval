from moteval.metrics.base import RustMetric, rust_metric_info


class JAndF(RustMetric):
    rust_name = "JAndF"
    fields, geometry = rust_metric_info("JAndF")
