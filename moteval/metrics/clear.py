from moteval.metrics.base import RustMetric, rust_metric_info


class CLEAR(RustMetric):
    rust_name = "CLEAR"
    fields, geometry = rust_metric_info("CLEAR")
