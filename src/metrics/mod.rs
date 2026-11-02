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

#[derive(Clone, Debug)]
pub enum Value {
    Float(f64),
    Array(Vec<f64>),
    Matrix(Matrix),
}

impl Value {
    pub fn as_slice(&self) -> &[f64] {
        match self {
            Self::Float(x) => std::slice::from_ref(x),
            Self::Array(values) => values,
            Self::Matrix(matrix) => matrix.data(),
        }
    }

    pub fn float(&self) -> f64 {
        match self {
            Self::Float(x) => *x,
            other => panic!("score value is not a float: {other:?}"),
        }
    }

    pub fn array(&self) -> &[f64] {
        match self {
            Self::Array(values) => values,
            other => panic!("score value is not a 1-D array: {other:?}"),
        }
    }

    pub fn matrix(&self) -> &Matrix {
        match self {
            Self::Matrix(matrix) => matrix,
            other => panic!("score value is not a 2-D array: {other:?}"),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::Float(_) => Vec::new(),
            Self::Array(values) => vec![values.len()],
            Self::Matrix(matrix) => vec![matrix.rows(), matrix.cols()],
        }
    }

    pub fn bits_eq(&self, other: &Self) -> bool {
        self.shape() == other.shape()
            && self
                .as_slice()
                .iter()
                .zip(other.as_slice())
                .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Scores {
    entries: Vec<(String, Value)>,
}

impl Scores {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn require(&self, field: &str) -> Result<&Value, MetricError> {
        self.get(field).ok_or_else(|| MetricError::Python {
            exception: "KeyError",
            message: crate::pyfmt::repr_str(field),
        })
    }

    pub fn float(&self, field: &str) -> f64 {
        match self.get(field) {
            Some(Value::Float(x)) => *x,
            other => panic!("score field {field:?} is not a float: {other:?}"),
        }
    }

    pub fn array(&self, field: &str) -> &[f64] {
        match self.get(field) {
            Some(Value::Array(values)) => values,
            other => panic!("score field {field:?} is not a 1-D array: {other:?}"),
        }
    }

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

#[derive(Clone, Debug, PartialEq)]
pub enum MetricError {
    Lsap(LsapError),
    Model(ModelError),
    Rle(RleError),
    Numeric(NumericError),
    NoResults {
        metric: &'static str,
    },
    Python {
        exception: &'static str,
        message: String,
    },
}

impl MetricError {
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

pub trait Metric: Send + Sync {
    fn name(&self) -> &'static str;

    fn fields(&self) -> &'static [&'static str];

    fn geometry(&self) -> Option<GeometryKind> {
        None
    }

    fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError>;

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError>;

    fn combine_classes_class_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError>;

    fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError>;
}

pub fn require_all(all_res: &[&Scores], field: &str) -> Result<(), MetricError> {
    all_res
        .iter()
        .try_for_each(|res| res.require(field).map(|_| ()))
}

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

/// TrackEval `_combine_sum`: CPython builtin `sum`, in order from the integer `0`.
fn plain_sum_values<'a>(values: impl Iterator<Item = Value> + 'a) -> Value {
    let mut acc = Value::Float(0.0);
    for value in values {
        acc = broadcast(&acc, &value, |a, b| a + b);
    }
    acc
}

pub fn combine_sum(all_res: &[&Scores], field: &str) -> Result<Value, MetricError> {
    let values = all_res
        .iter()
        .map(|res| res.require(field).cloned())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(plain_sum_values(values.into_iter()))
}

pub fn combine_weighted_av(
    all_res: &[&Scores],
    field: &str,
    comb_res: &Scores,
    weight_field: &str,
) -> Result<Value, MetricError> {
    let products = all_res
        .iter()
        .map(|res| {
            let value = res.require(field)?;
            let weight = res.require(weight_field)?;
            Ok(broadcast(value, weight, |a, b| a * b))
        })
        .collect::<Result<Vec<_>, MetricError>>()?;
    let weighted_sum = plain_sum_values(products.into_iter());
    let total_weight = comb_res
        .get(weight_field)
        .unwrap_or_else(|| panic!("combined field {weight_field:?} is missing"));
    let denominator = map_value(total_weight, |w| maximum(1.0, w));
    Ok(broadcast(&weighted_sum, &denominator, |a, b| a / b))
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
            combine_sum(&[&a, &b], "x").unwrap().as_slice()[0].to_bits(),
            0.0f64.to_bits()
        );
        let v = combine_sum(&[&a, &b], "v").unwrap();
        assert!(v.bits_eq(&Value::Array(vec![0.0, 3.0])));
    }

    #[test]
    fn weighted_average_uses_numpy_maximum() {
        let a = scores(&[("f", Value::Float(0.5)), ("w", Value::Float(2.0))]);
        let b = scores(&[("f", Value::Float(0.25)), ("w", Value::Float(4.0))]);
        let comb = scores(&[("w", Value::Float(6.0))]);
        let got = combine_weighted_av(&[&a, &b], "f", &comb, "w").unwrap();
        assert!(got.bits_eq(&Value::Float((0.5 * 2.0 + 0.25 * 4.0) / 6.0)));
        let comb_nan = scores(&[("w", Value::Float(f64::NAN))]);
        assert!(
            combine_weighted_av(&[&a, &b], "f", &comb_nan, "w")
                .unwrap()
                .as_slice()[0]
                .is_nan()
        );
        let comb_zero = scores(&[("w", Value::Float(0.0))]);
        let got = combine_weighted_av(&[&a, &b], "f", &comb_zero, "w").unwrap();
        assert!(got.bits_eq(&Value::Float(2.0)));
    }
}
