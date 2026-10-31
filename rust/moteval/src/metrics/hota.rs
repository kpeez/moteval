//! HOTA metric family over 19 alpha thresholds, plus HOTA(0), LocA(0), HOTALocA(0).
//!
//! The hota-trackmap package ports `moteval/metrics/hota.py` here, with all three combiners.

use crate::metrics::{Metric, MetricError, Scores};
use crate::model::SequenceData;

/// The public fields, in the Python `fields` order.
pub const FIELDS: &[&str] = &[
    "HOTA",
    "DetA",
    "AssA",
    "DetRe",
    "DetPr",
    "AssRe",
    "AssPr",
    "LocA",
    "OWTA",
    "HOTA_TP",
    "HOTA_FN",
    "HOTA_FP",
    "HOTA(0)",
    "LocA(0)",
    "HOTALocA(0)",
];

/// `moteval.metrics.hota.HOTA`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hota;

impl Metric for Hota {
    fn name(&self) -> &'static str {
        "HOTA"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn eval_sequence(&self, _data: &SequenceData) -> Result<Scores, MetricError> {
        todo!("hota-trackmap: port HOTA.eval_sequence")
    }

    fn combine_sequences(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("hota-trackmap: port HOTA.combine_sequences")
    }

    fn combine_classes_class_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("hota-trackmap: port HOTA.combine_classes_class_averaged")
    }

    fn combine_classes_det_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("hota-trackmap: port HOTA.combine_classes_det_averaged")
    }
}
