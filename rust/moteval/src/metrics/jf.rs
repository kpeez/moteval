//! J&F metric: DAVIS-style region similarity J and boundary accuracy F on masks.
//!
//! The rle-masks-jf package ports `moteval/metrics/jf.py` here, with all three combiners.

use crate::metrics::{Metric, MetricError, Scores};
use crate::model::{GeometryKind, SequenceData};

/// The public fields, in the Python `fields` order.
pub const FIELDS: &[&str] = &[
    "J-Mean",
    "J-Recall",
    "J-Decay",
    "F-Mean",
    "F-Recall",
    "F-Decay",
    "J&F",
    "num_gt_tracks",
];

/// `moteval.metrics.jf.JAndF`.
#[derive(Clone, Copy, Debug, Default)]
pub struct JAndF;

impl Metric for JAndF {
    fn name(&self) -> &'static str {
        "JAndF"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn geometry(&self) -> Option<GeometryKind> {
        Some(GeometryKind::Masks)
    }

    fn eval_sequence(&self, _data: &SequenceData) -> Result<Scores, MetricError> {
        todo!("rle-masks-jf: port JAndF.eval_sequence")
    }

    fn combine_sequences(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("rle-masks-jf: port JAndF.combine_sequences")
    }

    fn combine_classes_class_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("rle-masks-jf: port JAndF.combine_classes_class_averaged")
    }

    fn combine_classes_det_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("rle-masks-jf: port JAndF.combine_classes_det_averaged")
    }
}
