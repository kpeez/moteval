"""Count metric: detection and identity totals (no matching involved).

Implemented in ``src/metrics/count.rs``.
"""

from moteval.metrics.base import RustMetric, rust_metric_info


class Count(RustMetric):
    rust_name = "Count"
    fields, geometry = rust_metric_info("Count")
