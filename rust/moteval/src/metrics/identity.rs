//! Identity metric family: IDF1, IDR, IDP, IDTP, IDFN, IDFP.
//!
//! The clear-identity-count package ports `moteval/metrics/identity.py` here, with all three combiners.

use crate::metrics::{Metric, MetricError, Scores};
use crate::model::SequenceData;

/// The public fields, in the Python `fields` order.
pub const FIELDS: &[&str] = &["IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"];

/// `moteval.metrics.identity.Identity`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Metric for Identity {
    fn name(&self) -> &'static str {
        "Identity"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn eval_sequence(&self, _data: &SequenceData) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port Identity.eval_sequence")
    }

    fn combine_sequences(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port Identity.combine_sequences")
    }

    fn combine_classes_class_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port Identity.combine_classes_class_averaged")
    }

    fn combine_classes_det_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("clear-identity-count: port Identity.combine_classes_det_averaged")
    }
}
