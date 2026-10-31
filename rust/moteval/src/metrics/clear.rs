//! CLEAR (MOTChallenge) metric family: MOTA, MOTP, IDSW, MT/PT/ML, and friends.
//!
//! The clear-identity-count package ports `moteval/metrics/clear.py` here, with all three combiners.

use crate::metrics::{Metric, MetricError, Scores};
use crate::model::SequenceData;

/// The public fields, in the Python `fields` order.
pub const FIELDS: &[&str] = &[
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "sMOTA",
    "CLR_F1",
    "FP_per_frame",
    "MOTAL",
    "MOTP_sum",
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
    "CLR_Frames",
];

/// `moteval.metrics.clear.CLEAR`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Clear;

impl Metric for Clear {
    fn name(&self) -> &'static str {
        "CLEAR"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn eval_sequence(&self, _data: &SequenceData) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port CLEAR.eval_sequence")
    }

    fn combine_sequences(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port CLEAR.combine_sequences")
    }

    fn combine_classes_class_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port CLEAR.combine_classes_class_averaged")
    }

    fn combine_classes_det_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port CLEAR.combine_classes_det_averaged")
    }
}
