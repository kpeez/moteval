from moteval.metrics.base import RustMetric, rust_metric_info


class Count(RustMetric):
    rust_name = "Count"
    fields, geometry = rust_metric_info("Count")
