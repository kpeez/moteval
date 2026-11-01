from moteval.metrics.base import RustMetric, rust_metric_info


class HOTA(RustMetric):
    rust_name = "HOTA"
    fields, geometry = rust_metric_info("HOTA")
