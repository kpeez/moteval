//! TrackMAP metric: track-level average precision and recall over IoU thresholds, on all
//! tracks and on area and length subsets.
//!
//! The hota-trackmap package ports `moteval/metrics/track_map.py` here, with all three combiners.

use crate::metrics::{Metric, MetricError, Scores};
use crate::model::{GeometryKind, SequenceData};

/// The public fields, in the Python `fields` order.
pub const FIELDS: &[&str] = &[
    "AP_all",
    "AP_area_s",
    "AP_area_m",
    "AP_area_l",
    "AP_time_s",
    "AP_time_m",
    "AP_time_l",
    "AR_all",
    "AR_area_s",
    "AR_area_m",
    "AR_area_l",
    "AR_time_s",
    "AR_time_m",
    "AR_time_l",
];

/// `moteval.metrics.track_map.TrackMAP`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrackMap;

impl Metric for TrackMap {
    fn name(&self) -> &'static str {
        "TrackMAP"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn geometry(&self) -> Option<GeometryKind> {
        Some(GeometryKind::Boxes)
    }

    fn eval_sequence(&self, _data: &SequenceData) -> Result<Scores, MetricError> {
        todo!("hota-trackmap: port TrackMAP.eval_sequence")
    }

    fn combine_sequences(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("hota-trackmap: port TrackMAP.combine_sequences")
    }

    fn combine_classes_class_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("hota-trackmap: port TrackMAP.combine_classes_class_averaged")
    }

    fn combine_classes_det_averaged(&self, _all_res: &[&Scores]) -> Result<Scores, MetricError> {
        todo!("hota-trackmap: port TrackMAP.combine_classes_det_averaged")
    }
}
