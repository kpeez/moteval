//! The `Metric` trait and the score container every metric shares, ported from
//! `moteval/metrics/base.py`.
//!
//! A metric declares its `fields` and the `geometry` it reads, scores one `SequenceData`
//! at a time, and combines per-sequence results and per-class results (class-averaged and
//! detection-averaged).

pub mod clear;
pub mod count;
pub mod hota;
pub mod identity;
pub mod jf;
pub mod track_map;

use std::fmt;

use crate::lsap::LsapError;
use crate::model::{GeometryKind, Matrix, ModelError, SequenceData};
use crate::numeric::{NumericError, maximum};
use crate::rle::RleError;

pub use clear::Clear;
pub use count::Count;
pub use hota::Hota;
pub use identity::Identity;
pub use jf::JAndF;
pub use track_map::TrackMap;

/// One score value: Python's `float | np.ndarray`.
///
/// Integer counts (HOTA's `HOTA_TP`, CLEAR's `IDSW`) are floats in Python too. Private
/// state that Python holds as `bool` or `int` arrays (TrackMAP's `_dt_ignore_*`,
/// `_gt_ignore_*`) is stored here as exact `0.0`/`1.0`/integer floats.
#[derive(Clone, Debug)]
pub enum Value {
    /// A Python `float` or `np.float64`.
    Float(f64),
    /// A 1-D array (HOTA's 19 alphas, TrackMAP's 10 IoU thresholds).
    Array(Vec<f64>),
    /// A 2-D C-order array (TrackMAP's per-threshold match state).
    Matrix(Matrix),
}

impl Value {
    /// The values in C order: one for `Float`, every element otherwise.
    pub fn as_slice(&self) -> &[f64] {
        match self {
            Self::Float(x) => std::slice::from_ref(x),
            Self::Array(values) => values,
            Self::Matrix(matrix) => matrix.data(),
        }
    }

    /// The numpy shape: `[]`, `[n]`, or `[rows, cols]`.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::Float(_) => Vec::new(),
            Self::Array(values) => vec![values.len()],
            Self::Matrix(matrix) => vec![matrix.rows(), matrix.cols()],
        }
    }

    /// Bit-for-bit equality: same shape, and every element has the same bits, except that
    /// any NaN equals any NaN (`np.array_equal(..., equal_nan=True)` on exact values).
    pub fn bits_eq(&self, other: &Self) -> bool {
        self.shape() == other.shape()
            && self
                .as_slice()
                .iter()
                .zip(other.as_slice())
                .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
    }
}

/// One metric's scores: an insertion-ordered map from field name to value, like the
/// Python `dict` it ports.
///
/// Order is output order: the JSON, CSV, and table writers emit fields in the order the
/// metric inserted them, so a port must insert in the order its Python dict was built.
/// Re-inserting an existing field keeps its position, as Python dict assignment does.
/// Fields that start with `_` are private state for the combiners (TrackMAP's match
/// arrays and `_num_dt_*` weights); `evaluate` reports only `Metric::fields`.
#[derive(Clone, Debug, Default)]
pub struct Scores {
    entries: Vec<(String, Value)>,
}

impl Scores {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `field`, keeping its position if it is already present.
    pub fn insert(&mut self, field: impl Into<String>, value: Value) {
        let field = field.into();
        match self.entries.iter_mut().find(|(name, _)| *name == field) {
            Some((_, slot)) => *slot = value,
            None => self.entries.push((field, value)),
        }
    }

    pub fn get(&self, field: &str) -> Option<&Value> {
        self.entries
            .iter()
            .find(|(name, _)| name == field)
            .map(|(_, value)| value)
    }

    /// The scalar `field`. Panics if it is missing or not a `Float`: a metric reading its
    /// own state has a bug if this fails.
    pub fn float(&self, field: &str) -> f64 {
        match self.get(field) {
            Some(Value::Float(x)) => *x,
            other => panic!("score field {field:?} is not a float: {other:?}"),
        }
    }

    /// The 1-D array `field`. Panics if it is missing or not an `Array`.
    pub fn array(&self, field: &str) -> &[f64] {
        match self.get(field) {
            Some(Value::Array(values)) => values,
            other => panic!("score field {field:?} is not a 1-D array: {other:?}"),
        }
    }

    /// The 2-D array `field`. Panics if it is missing or not a `Matrix`.
    pub fn matrix(&self, field: &str) -> &Matrix {
        match self.get(field) {
            Some(Value::Matrix(matrix)) => matrix,
            other => panic!("score field {field:?} is not a 2-D array: {other:?}"),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `eval._declared`: only the entries named in `fields`, in this map's order (not the
    /// order of `fields`).
    pub fn declared(&self, fields: &[&str]) -> Scores {
        Scores {
            entries: self
                .entries
                .iter()
                .filter(|(name, _)| fields.contains(&name.as_str()))
                .cloned()
                .collect(),
        }
    }
}

/// Errors a metric can raise. Python raises these from inside scipy, pycocotools, or the
/// data model; the Python layer maps each to `ValueError` with the same text.
#[derive(Clone, Debug, PartialEq)]
pub enum MetricError {
    Lsap(LsapError),
    Model(ModelError),
    Rle(RleError),
    Numeric(NumericError),
    /// A combiner called with no results. Python fails here too, but each metric fails
    /// differently (HOTA: `IndexError`; TrackMAP: numpy's "need at least one array to
    /// concatenate") or returns integer zeros (Count, CLEAR, Identity), so `evaluate`
    /// decides what an empty dataset means.
    NoResults {
        metric: &'static str,
    },
    /// An error Python or numpy raises inside a metric, with its class (`"IndexError"`,
    /// `"ValueError"`) and message text. J&F raises these on sequences TrackEval cannot
    /// score either, such as no GT ids and two or more predicted ids.
    Python {
        exception: &'static str,
        message: String,
    },
}

impl MetricError {
    /// The Python exception class this error stands for.
    pub fn python_exception(&self) -> &'static str {
        match self {
            Self::Python { exception, .. } => exception,
            _ => "ValueError",
        }
    }
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lsap(err) => err.fmt(f),
            Self::Model(err) => err.fmt(f),
            Self::Rle(err) => err.fmt(f),
            Self::Numeric(err) => err.fmt(f),
            Self::NoResults { metric } => write!(f, "{metric}: no results to combine"),
            Self::Python { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for MetricError {}

impl From<LsapError> for MetricError {
    fn from(err: LsapError) -> Self {
        Self::Lsap(err)
    }
}

impl From<ModelError> for MetricError {
    fn from(err: ModelError) -> Self {
        Self::Model(err)
    }
}

impl From<RleError> for MetricError {
    fn from(err: RleError) -> Self {
        Self::Rle(err)
    }
}

impl From<NumericError> for MetricError {
    fn from(err: NumericError) -> Self {
        Self::Numeric(err)
    }
}

/// One metric family, mirroring the `Metric` ABC.
///
/// The combiners take results in Python's dict order: sequences in dataset order for
/// `combine_sequences`, classes in `Protocol::eval_classes` order for the class
/// combiners. Their inputs are the raw state `eval_sequence` and `combine_sequences`
/// return, private fields included.
pub trait Metric: Send + Sync {
    /// The Python class name (`type(metric).__name__`): the metric's key in results.
    fn name(&self) -> &'static str;

    /// The public result fields, in the Python `fields` tuple order.
    fn fields(&self) -> &'static [&'static str];

    /// The geometry `eval_sequence` reads besides the precomputed similarity: boxes
    /// (TrackMAP), masks (J&F), or none.
    fn geometry(&self) -> Option<GeometryKind> {
        None
    }

    fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError>;

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError>;

    fn combine_classes_class_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError>;

    fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError>;
}

fn values_of<'a>(all_res: &'a [&'a Scores], field: &str) -> impl Iterator<Item = &'a Value> {
    all_res.iter().map(move |res| {
        res.get(field)
            .unwrap_or_else(|| panic!("score field {field:?} is missing"))
    })
}

/// Elementwise `a op b` with numpy broadcasting between a scalar and an array (or two
/// same-shape values).
fn broadcast(a: &Value, b: &Value, op: impl Fn(f64, f64) -> f64) -> Value {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => Value::Float(op(*x, *y)),
        (Value::Float(x), other) => map_value(other, |y| op(*x, y)),
        (other, Value::Float(y)) => map_value(other, |x| op(x, *y)),
        (x, y) => {
            assert_eq!(x.shape(), y.shape(), "score values differ in shape");
            let data: Vec<f64> = x
                .as_slice()
                .iter()
                .zip(y.as_slice())
                .map(|(&p, &q)| op(p, q))
                .collect();
            match x {
                Value::Matrix(m) => {
                    Value::Matrix(Matrix::new(m.rows(), m.cols(), data).expect("same shape"))
                }
                _ => Value::Array(data),
            }
        }
    }
}

fn map_value(value: &Value, op: impl Fn(f64) -> f64) -> Value {
    match value {
        Value::Float(x) => Value::Float(op(*x)),
        Value::Array(values) => Value::Array(values.iter().map(|&x| op(x)).collect()),
        Value::Matrix(m) => Value::Matrix(
            Matrix::new(
                m.rows(),
                m.cols(),
                m.data().iter().map(|&x| op(x)).collect(),
            )
            .expect("same shape"),
        ),
    }
}

/// `_plain_sum` over values: in order, starting from the integer `0` (an exact `+0.0`),
/// elementwise for arrays.
fn plain_sum_values<'a>(values: impl Iterator<Item = Value> + 'a) -> Value {
    let mut acc = Value::Float(0.0);
    for value in values {
        acc = broadcast(&acc, &value, |a, b| a + b);
    }
    acc
}

/// `Metric._combine_sum(all_res, field)`: `functools.reduce(operator.add, values, 0)`,
/// so the sum runs in order from `+0.0`, elementwise for arrays.
///
/// With no results Python returns the integer `0`; this returns `Float(0.0)`, and each
/// metric must decide (see `MetricError::NoResults`).
pub fn combine_sum(all_res: &[&Scores], field: &str) -> Value {
    plain_sum_values(values_of(all_res, field).cloned())
}

/// `Metric._combine_weighted_av(all_res, field, comb_res, weight_field)`:
/// `_plain_sum(res[field] * res[weight_field]) / np.maximum(1.0, comb_res[weight_field])`.
///
/// The products and the sum are elementwise for arrays; `np.maximum` propagates NaN.
pub fn combine_weighted_av(
    all_res: &[&Scores],
    field: &str,
    comb_res: &Scores,
    weight_field: &str,
) -> Value {
    let products = all_res.iter().map(|res| {
        let value = res
            .get(field)
            .unwrap_or_else(|| panic!("score field {field:?} is missing"));
        let weight = res
            .get(weight_field)
            .unwrap_or_else(|| panic!("score field {weight_field:?} is missing"));
        broadcast(value, weight, |a, b| a * b)
    });
    let weighted_sum = plain_sum_values(products);
    let total_weight = comb_res
        .get(weight_field)
        .unwrap_or_else(|| panic!("combined field {weight_field:?} is missing"));
    let denominator = map_value(total_weight, |w| maximum(1.0, w));
    broadcast(&weighted_sum, &denominator, |a, b| a / b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(pairs: &[(&str, Value)]) -> Scores {
        let mut s = Scores::new();
        for (name, value) in pairs {
            s.insert(*name, value.clone());
        }
        s
    }

    #[test]
    fn insert_keeps_position_and_declared_keeps_map_order() {
        let mut s = scores(&[
            ("b", Value::Float(1.0)),
            ("a", Value::Float(2.0)),
            ("_private", Value::Float(3.0)),
        ]);
        s.insert("b", Value::Float(4.0));
        let names: Vec<&str> = s.iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["b", "a", "_private"]);
        assert_eq!(s.float("b"), 4.0);
        let declared = s.declared(&["a", "b"]);
        let names: Vec<&str> = declared.iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["b", "a"]);
    }

    #[test]
    fn combine_sum_starts_from_positive_zero() {
        let a = scores(&[
            ("x", Value::Float(-0.0)),
            ("v", Value::Array(vec![-0.0, 1.0])),
        ]);
        let b = scores(&[
            ("x", Value::Float(-0.0)),
            ("v", Value::Array(vec![-0.0, 2.0])),
        ]);
        assert_eq!(
            combine_sum(&[&a, &b], "x").as_slice()[0].to_bits(),
            0.0f64.to_bits()
        );
        let v = combine_sum(&[&a, &b], "v");
        assert!(v.bits_eq(&Value::Array(vec![0.0, 3.0])));
    }

    #[test]
    fn weighted_average_uses_numpy_maximum() {
        let a = scores(&[("f", Value::Float(0.5)), ("w", Value::Float(2.0))]);
        let b = scores(&[("f", Value::Float(0.25)), ("w", Value::Float(4.0))]);
        let comb = scores(&[("w", Value::Float(6.0))]);
        let got = combine_weighted_av(&[&a, &b], "f", &comb, "w");
        assert!(got.bits_eq(&Value::Float((0.5 * 2.0 + 0.25 * 4.0) / 6.0)));
        let comb_nan = scores(&[("w", Value::Float(f64::NAN))]);
        assert!(combine_weighted_av(&[&a, &b], "f", &comb_nan, "w").as_slice()[0].is_nan());
        let comb_zero = scores(&[("w", Value::Float(0.0))]);
        let got = combine_weighted_av(&[&a, &b], "f", &comb_zero, "w");
        assert!(got.bits_eq(&Value::Float(2.0)));
    }
}
