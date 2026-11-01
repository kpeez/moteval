use crate::metrics::{Scores, Value};
use crate::{numeric, pyfmt};

#[derive(Clone, Debug, Default)]
pub struct MetricScores {
    entries: Vec<(String, Scores)>,
}

impl MetricScores {
    pub fn new() -> Self {
        Self::default()
    }

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

#[derive(Clone, Debug, Default)]
pub struct ClassResult {
    pub per_sequence: Vec<(String, MetricScores)>,
    pub combined: MetricScores,
}

#[derive(Clone, Debug, Default)]
pub struct EvaluationResult {
    pub per_sequence: Vec<(String, MetricScores)>,
    pub combined: MetricScores,
    pub per_class: Vec<(i64, ClassResult)>,
    pub class_averaged: MetricScores,
    pub det_averaged: MetricScores,
}

impl EvaluationResult {
    pub fn is_multi_class(&self) -> bool {
        !self.per_class.is_empty()
    }
}

pub const COMBINED: &str = "COMBINED";
pub const CLASS_AVERAGED: &str = "class_averaged";
pub const DET_AVERAGED: &str = "det_averaged";

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

pub fn row_label_names(result: &EvaluationResult) -> &'static [&'static str] {
    if result.is_multi_class() {
        &["class", "seq"]
    } else {
        &["seq"]
    }
}

pub fn scalar_value(value: &Value) -> f64 {
    match value {
        Value::Float(x) => *x,
        Value::Array(_) | Value::Matrix(_) => numeric::mean(value.as_slice()),
    }
}

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

fn json_scores(scores: &MetricScores, out: &mut String) {
    json_object(scores.iter(), out, |fields: &Scores, out| {
        json_object(fields.iter(), out, json_value);
    });
}

fn json_per_sequence(per_sequence: &[(String, MetricScores)], out: &mut String) {
    json_object(
        per_sequence
            .iter()
            .map(|(name, scores)| (name.as_str(), scores)),
        out,
        json_scores,
    );
}

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
