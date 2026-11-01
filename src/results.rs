//! Typed evaluation results and their stable JSON and CSV export shapes, ported from
//! `moteval/results.py`.
//!
//! A single-class run fills `per_sequence` and `combined`. A multi-class run fills
//! `per_class`, `class_averaged` and `det_averaged` instead, and leaves `per_sequence` and
//! `combined` empty. Every mapping keeps Python's insertion order, which is the export
//! order.
//!
//! The JSON and CSV bytes equal what Python writes (`tests/integration/exports.rs`).

use crate::metrics::{Scores, Value};
use crate::{numeric, pyfmt};

/// Metric name -> that metric's declared scores, in metric order.
#[derive(Clone, Debug, Default)]
pub struct MetricScores {
    entries: Vec<(String, Scores)>,
}

impl MetricScores {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `metric`, keeping its position if it is already present.
    pub fn insert(&mut self, metric: impl Into<String>, scores: Scores) {
        let metric = metric.into();
        match self.entries.iter_mut().find(|(name, _)| *name == metric) {
            Some((_, slot)) => *slot = scores,
            None => self.entries.push((metric, scores)),
        }
    }

    pub fn get(&self, metric: &str) -> Option<&Scores> {
        self.entries
            .iter()
            .find(|(name, _)| name == metric)
            .map(|(_, scores)| scores)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Scores)> {
        self.entries
            .iter()
            .map(|(name, scores)| (name.as_str(), scores))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Scores for one class of a multi-class run, shaped like a single-class run.
#[derive(Clone, Debug, Default)]
pub struct ClassResult {
    /// Sequence name -> scores, in dataset order.
    pub per_sequence: Vec<(String, MetricScores)>,
    pub combined: MetricScores,
}

/// Scores for one evaluation run.
#[derive(Clone, Debug, Default)]
pub struct EvaluationResult {
    /// Sequence name -> scores, in dataset order. Empty for a multi-class run.
    pub per_sequence: Vec<(String, MetricScores)>,
    /// Empty for a multi-class run.
    pub combined: MetricScores,
    /// Class id -> that class's scores, in `eval_classes` order. Empty for a
    /// single-class run.
    pub per_class: Vec<(i64, ClassResult)>,
    pub class_averaged: MetricScores,
    pub det_averaged: MetricScores,
}

impl EvaluationResult {
    pub fn is_multi_class(&self) -> bool {
        !self.per_class.is_empty()
    }
}

/// The row label of each sequence's combined scores.
pub const COMBINED: &str = "COMBINED";
/// The class label of the class-averaged row of a multi-class run.
pub const CLASS_AVERAGED: &str = "class_averaged";
/// The class label of the detection-averaged row of a multi-class run.
pub const DET_AVERAGED: &str = "det_averaged";

/// `iter_scored_rows`: `(row labels, scores)` in export order.
///
/// Labels are `[seq]` for a single-class run. A multi-class run gives `[class, seq]`: each
/// class's sequences and then its `COMBINED` row, class by class, then the
/// `class_averaged` and `det_averaged` rows with `seq = COMBINED`.
pub fn scored_rows(result: &EvaluationResult) -> Vec<(Vec<String>, &MetricScores)> {
    let mut rows = Vec::new();
    if !result.is_multi_class() {
        for (sequence, scores) in &result.per_sequence {
            rows.push((vec![sequence.clone()], scores));
        }
        rows.push((vec![COMBINED.to_string()], &result.combined));
        return rows;
    }
    for (cls_id, class_result) in &result.per_class {
        for (sequence, scores) in &class_result.per_sequence {
            rows.push((vec![cls_id.to_string(), sequence.clone()], scores));
        }
        rows.push((
            vec![cls_id.to_string(), COMBINED.to_string()],
            &class_result.combined,
        ));
    }
    rows.push((
        vec![CLASS_AVERAGED.to_string(), COMBINED.to_string()],
        &result.class_averaged,
    ));
    rows.push((
        vec![DET_AVERAGED.to_string(), COMBINED.to_string()],
        &result.det_averaged,
    ));
    rows
}

/// `row_label_names`: `["seq"]`, or `["class", "seq"]` for a multi-class run.
pub fn row_label_names(result: &EvaluationResult) -> &'static [&'static str] {
    if result.is_multi_class() {
        &["class", "seq"]
    } else {
        &["seq"]
    }
}

/// `np.mean(value)` for an array field, the value itself for a scalar: the one number the
/// CSV export and the CLI table show for a field. numpy's mean is pairwise over the flat
/// C-order data, for 1-D and 2-D arrays alike.
pub fn scalar_value(value: &Value) -> f64 {
    match value {
        Value::Float(x) => *x,
        Value::Array(_) | Value::Matrix(_) => numeric::mean(value.as_slice()),
    }
}

/// A value as `json.dumps` writes `ndarray.tolist()` or a float.
fn json_value(value: &Value, out: &mut String) {
    let list = |values: &[f64], out: &mut String| {
        out.push('[');
        for (index, &x) in values.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            out.push_str(&pyfmt::json_float(x));
        }
        out.push(']');
    };
    match value {
        Value::Float(x) => out.push_str(&pyfmt::json_float(*x)),
        Value::Array(values) => list(values, out),
        Value::Matrix(matrix) => {
            out.push('[');
            for row in 0..matrix.rows() {
                if row > 0 {
                    out.push_str(", ");
                }
                list(matrix.row(row), out);
            }
            out.push(']');
        }
    }
}

/// A JSON object of `entries`, written by `write_value`, as `json.dumps` lays it out.
fn json_object<'a, T: 'a>(
    entries: impl IntoIterator<Item = (&'a str, T)>,
    out: &mut String,
    mut write_value: impl FnMut(T, &mut String),
) {
    out.push('{');
    for (index, (key, value)) in entries.into_iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&pyfmt::json_str(key));
        out.push_str(": ");
        write_value(value, out);
    }
    out.push('}');
}

/// `_json_scores`: metric -> field -> value.
fn json_scores(scores: &MetricScores, out: &mut String) {
    json_object(scores.iter(), out, |fields: &Scores, out| {
        json_object(fields.iter(), out, json_value);
    });
}

/// `_json_per_sequence`: sequence -> metric -> field -> value.
fn json_per_sequence(per_sequence: &[(String, MetricScores)], out: &mut String) {
    json_object(
        per_sequence
            .iter()
            .map(|(name, scores)| (name.as_str(), scores)),
        out,
        json_scores,
    );
}

/// `json.dumps(to_json_dict(result, dataset=dataset, split=split))`: the stable JSON
/// export, byte for byte.
///
/// Keys: `dataset`, `split`, `per_sequence`, `combined`; a multi-class run adds
/// `per_class` (class id as a string -> `per_sequence` + `combined`), `class_averaged`,
/// and `det_averaged`.
pub fn to_json(result: &EvaluationResult, dataset: &str, split: &str) -> String {
    let mut out = String::new();
    out.push_str("{\"dataset\": ");
    out.push_str(&pyfmt::json_str(dataset));
    out.push_str(", \"split\": ");
    out.push_str(&pyfmt::json_str(split));
    out.push_str(", \"per_sequence\": ");
    json_per_sequence(&result.per_sequence, &mut out);
    out.push_str(", \"combined\": ");
    json_scores(&result.combined, &mut out);
    if result.is_multi_class() {
        out.push_str(", \"per_class\": ");
        let classes: Vec<(String, &ClassResult)> = result
            .per_class
            .iter()
            .map(|(cls_id, class_result)| (cls_id.to_string(), class_result))
            .collect();
        json_object(
            classes
                .iter()
                .map(|(key, class_result)| (key.as_str(), *class_result)),
            &mut out,
            |class_result: &ClassResult, out| {
                out.push_str("{\"per_sequence\": ");
                json_per_sequence(&class_result.per_sequence, out);
                out.push_str(", \"combined\": ");
                json_scores(&class_result.combined, out);
                out.push('}');
            },
        );
        out.push_str(", \"class_averaged\": ");
        json_scores(&result.class_averaged, &mut out);
        out.push_str(", \"det_averaged\": ");
        json_scores(&result.det_averaged, &mut out);
    }
    out.push('}');
    out
}

/// The long-form CSV file `cli._write_csv` writes, byte for byte: `csv_header`, then one
/// row per field of `iter_csv_rows`, array fields reduced to their mean.
pub fn to_csv(result: &EvaluationResult) -> String {
    let labels = row_label_names(result);
    let mut header: Vec<&str> = labels.to_vec();
    header.extend(["metric", "field", "value"]);
    let mut out = pyfmt::csv_row(&header);
    for (row_labels, scores) in scored_rows(result) {
        for (metric, fields) in scores.iter() {
            for (field, value) in fields.iter() {
                let mut row = row_labels.clone();
                row.push(metric.to_string());
                row.push(field.to_string());
                row.push(pyfmt::repr_float(scalar_value(value)));
                out.push_str(&pyfmt::csv_row(&row));
            }
        }
    }
    out
}
