//! Top-level evaluation, ported from `moteval/eval.py`: score a loaded `MOTDataset`
//! against `<seq>.txt` prediction files.
//!
//! Each class in `protocol.eval_classes` is scored on its own: one `SequenceData` view per
//! sequence, every metric, then the sequences combined for that class. A multi-class run
//! also combines the classes, class-averaged and detection-averaged, from the raw
//! per-class state. Results hold only each metric's declared `fields`.
//!
//! The results-cli-bindings package fills the body.

use std::fmt;
use std::path::Path;

use crate::convert::ConvertError;
use crate::formats::FormatError;
use crate::metrics::{Metric, MetricError};
use crate::model::MOTDataset;
use crate::results::EvaluationResult;

/// Errors `evaluate` raises before or while scoring. Messages match the Python text; the
/// Python layer raises each as `ValueError`.
#[derive(Debug)]
pub enum EvalError {
    /// `duplicate metric classes in metrics: {names}` (sorted, comma-separated).
    DuplicateMetrics(Vec<String>),
    /// `protocol {name!r} declares no eval_classes`.
    NoEvalClasses {
        protocol: String,
    },
    /// `protocol {name!r} repeats a class in {classes}` (`classes` as a Python tuple).
    RepeatedClass {
        protocol: String,
        classes: Vec<i64>,
    },
    /// `{metric} reads {geometry}, but sequence {seq!r} of dataset {dataset!r} holds {kind}`.
    GeometryMismatch {
        metric: String,
        reads: &'static str,
        sequence: String,
        dataset: String,
        holds: &'static str,
    },
    /// `prediction directory {dir} has no <seq>.txt file for {n} sequence(s): {names}
    /// (an empty file means no predictions)`.
    MissingPredictions {
        dir: String,
        missing: Vec<String>,
    },
    Format(FormatError),
    Convert(ConvertError),
    Metric(MetricError),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::pyfmt::repr_str;
        match self {
            Self::DuplicateMetrics(names) => write!(
                f,
                "duplicate metric classes in metrics: {}",
                names.join(", ")
            ),
            Self::NoEvalClasses { protocol } => {
                write!(
                    f,
                    "protocol {} declares no eval_classes",
                    repr_str(protocol)
                )
            }
            Self::RepeatedClass { protocol, classes } => {
                let items: Vec<String> = classes.iter().map(i64::to_string).collect();
                let tuple = if items.len() == 1 {
                    format!("({},)", items[0])
                } else {
                    format!("({})", items.join(", "))
                };
                write!(
                    f,
                    "protocol {} repeats a class in {tuple}",
                    repr_str(protocol)
                )
            }
            Self::GeometryMismatch {
                metric,
                reads,
                sequence,
                dataset,
                holds,
            } => write!(
                f,
                "{metric} reads {reads}, but sequence {} of dataset {} holds {holds}",
                repr_str(sequence),
                repr_str(dataset)
            ),
            Self::MissingPredictions { dir, missing } => write!(
                f,
                "prediction directory {dir} has no <seq>.txt file for {} sequence(s): {} \
                 (an empty file means no predictions)",
                missing.len(),
                missing.join(", ")
            ),
            Self::Format(err) => err.fmt(f),
            Self::Convert(err) => err.fmt(f),
            Self::Metric(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for EvalError {}

/// `evaluate(dataset, predictions, metrics)`.
pub fn evaluate(
    _dataset: &MOTDataset,
    _predictions: &Path,
    _metrics: &[Box<dyn Metric>],
) -> Result<EvaluationResult, EvalError> {
    todo!("results-cli-bindings: port moteval.eval.evaluate")
}
