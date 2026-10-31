//! Count metric: detection and identity totals (no matching), ported from
//! `moteval/metrics/count.py`.

use crate::metrics::{Metric, MetricError, Scores, Value, combine_sum};
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

    fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError> {
        let mut res = Scores::new();
        res.insert("Dets", Value::Float(data.num_pred_dets() as f64));
        res.insert("GT_Dets", Value::Float(data.num_gt_dets() as f64));
        res.insert("IDs", Value::Float(data.num_pred_ids() as f64));
        res.insert("GT_IDs", Value::Float(data.num_gt_ids() as f64));
        Ok(res)
    }

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        sum_every_field(all_res)
    }

    fn combine_classes_class_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        sum_every_field(all_res)
    }

    fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        sum_every_field(all_res)
    }
}

/// Every combiner: `{f: self._combine_sum(all_res, f) for f in self.fields}`.
fn sum_every_field(all_res: &[&Scores]) -> Result<Scores, MetricError> {
    if all_res.is_empty() {
        return Err(MetricError::NoResults { metric: "Count" });
    }
    let mut res = Scores::new();
    for &field in FIELDS {
        res.insert(field, combine_sum(all_res, field));
    }
    Ok(res)
}
