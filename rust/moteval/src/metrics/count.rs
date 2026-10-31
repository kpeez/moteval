//! Count metric: detection and identity totals (no matching).
//!
//! The clear-identity-count package ports `moteval/metrics/count.py` here, with all three combiners.

use crate::metrics::{Metric, MetricError, Scores};
use crate::model::SequenceData;

/// The public fields, in the Python `fields` order.
pub const FIELDS: &[&str] = &["Dets", "GT_Dets", "IDs", "GT_IDs"];

/// `moteval.metrics.count.Count`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Count;

impl Metric for Count {
    fn name(&self) -> &'static str {
        "Count"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn eval_sequence(&self, _data: &SequenceData) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port Count.eval_sequence")
    }

    fn combine_sequences(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port Count.combine_sequences")
    }

    fn combine_classes_class_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port Count.combine_classes_class_averaged")
    }

    fn combine_classes_det_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port Count.combine_classes_det_averaged")
    }
}
