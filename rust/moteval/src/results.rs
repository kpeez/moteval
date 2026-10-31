//! Typed evaluation results and their stable JSON and CSV export shapes, ported from
//! `moteval/results.py`.
//!
//! A single-class run fills `per_sequence` and `combined`. A multi-class run fills
//! `per_class`, `class_averaged` and `det_averaged` instead, and leaves `per_sequence` and
//! `combined` empty. Every mapping keeps Python's insertion order, which is the export
//! order.
//!
//! The results-cli-bindings package fills the export functions; the JSON and CSV bytes
//! must equal what Python writes.

use crate::metrics::Scores;

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

/// `json.dumps(to_json_dict(result, dataset=dataset, split=split))`.
pub fn to_json(_result: &EvaluationResult, _dataset: &str, _split: &str) -> String {
    todo!("results-cli-bindings: port moteval.results.to_json_dict and json.dumps")
}

/// The long-form CSV file `cli._write_csv` writes: `csv_header` then `iter_csv_rows`.
pub fn to_csv(_result: &EvaluationResult) -> String {
    todo!("results-cli-bindings: port moteval.results.iter_csv_rows and csv.writer")
}
