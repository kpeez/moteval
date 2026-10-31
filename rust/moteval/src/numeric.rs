//! numpy and Python float semantics, bit for bit.
//!
//! Every reduction, range, and rounding the metrics need lives here, so each Python call
//! site maps to one function with the same summation order. The reference is numpy 2.5.1
//! on CPython 3.14; `rust/testdata/numeric.json` (from `rust/scripts/dump_numeric.py`)
//! holds numpy's outputs as `u64` bit patterns, and `tests/numeric.rs` checks every
//! function against them.
//!
//! | Python call site | Function |
//! |---|---|
//! | `np.sum(a)`, `a.sum()` on contiguous `float64` (1-D, or 2-D with no axis) | [`sum`] |
//! | `np.mean(a)` on the same | [`mean`] |
//! | `np.sum(m, axis=0)`, `np.mean(m, axis=0)` on C-order 2-D | [`sum_axis0`], [`mean_axis0`] |
//! | `np.sum(m, axis=-1)`, `np.mean(m, axis=-1)` (last axis) | [`sum_last_axis`], [`mean_last_axis`] |
//! | `np.nanmean(a)` on `float64` | [`nanmean`] |
//! | `np.cumsum(a)` | [`cumsum`] |
//! | `np.average(a, weights=w)` (1-D) | [`average`] |
//! | builtin `sum(...)` over `np.float64`, `functools.reduce(operator.add, xs, 0)` | [`plain_sum`] |
//! | `np.arange(start, stop, step)` (floats) | [`arange`] |
//! | `np.linspace(start, stop, num, endpoint)` | [`linspace`] |
//! | `np.round(x)` / `np.rint(x)` | [`round_half_even`] |
//! | `np.maximum`, `np.minimum` | [`maximum`], [`minimum`] |
//! | builtin `max(a, b)`, `min(a, b)` | [`py_max`], [`py_min`] |
//! | `np.finfo(float).eps`, `np.spacing(1)` | [`EPS`] |

use std::fmt;

/// `np.finfo(float).eps` and `np.spacing(1)`: `2**-52`.
pub const EPS: f64 = f64::EPSILON;

/// Errors from the numeric helpers. Messages match numpy's text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumericError {
    /// `np.arange` with a zero step (Python `ZeroDivisionError`).
    ArangeZeroStep,
    /// `np.arange` whose length is NaN (`ValueError`).
    ArangeLength,
    /// `np.arange` whose length does not fit an index (numpy: `ValueError`).
    ArangeSize,
    /// `np.average` whose weights sum to zero (`ZeroDivisionError`).
    WeightsSumToZero,
    /// `np.average` with values and weights of different lengths.
    ShapeMismatch,
}

impl fmt::Display for NumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArangeZeroStep => f.write_str("division by zero"),
            Self::ArangeLength => f.write_str("arange: cannot compute length"),
            Self::ArangeSize => f.write_str("Maximum allowed size exceeded"),
            Self::WeightsSumToZero => f.write_str("Weights sum to zero, can't be normalized"),
            Self::ShapeMismatch => {
                f.write_str("Length of weights not compatible with specified axis.")
            }
        }
    }
}

impl std::error::Error for NumericError {}

/// numpy's `pairwise_sum` (numpy 2.5.1 `loops_utils.h.src`), with no initial value.
///
/// Below 8 elements: a plain loop from `-0.0`. Up to 128: eight accumulators, combined as
/// `((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))`, then the remainder in order. Above
/// 128: split at `n / 2` rounded down to a multiple of 8 and recurse.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            r[0] += a[i];
            r[1] += a[i + 1];
            r[2] += a[i + 2];
            r[3] += a[i + 3];
            r[4] += a[i + 4];
            r[5] += a[i + 5];
            r[6] += a[i + 6];
            r[7] += a[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum(a)` over a contiguous `float64` array: the identity `0.0` plus the pairwise
/// sum of every element. A C-order 2-D array summed with no axis reduces its flat data
/// the same way, with no blocking (checked to 300,001 elements).
pub fn sum(a: &[f64]) -> f64 {
    0.0 + pairwise_sum(a)
}

/// `np.mean(a)`: `sum(a) / n`. An empty array gives NaN, as numpy does (with a warning).
pub fn mean(a: &[f64]) -> f64 {
    sum(a) / a.len() as f64
}

/// `np.sum(m, axis=0)` over a C-order `rows x cols` array: row by row, in order, from
/// zeros. Not pairwise, because numpy's inner loop runs along the contiguous last axis.
///
/// A single column (`cols == 1`) is the exception: numpy drops the length-1 axis, the
/// reduced axis is then contiguous, and it reduces like [`sum`] (pairwise).
pub fn sum_axis0(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    assert_eq!(data.len(), rows * cols, "data is not rows x cols");
    if cols == 1 {
        return vec![sum(data)];
    }
    let mut acc = vec![0.0; cols];
    for row in data.chunks_exact(cols.max(1)).take(rows) {
        for (a, &x) in acc.iter_mut().zip(row) {
            *a += x;
        }
    }
    acc
}

/// `np.mean(m, axis=0)` over a C-order `rows x cols` array: [`sum_axis0`] divided by
/// `rows`.
pub fn mean_axis0(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    sum_axis0(data, rows, cols)
        .into_iter()
        .map(|s| s / rows as f64)
        .collect()
}

/// `np.sum(m, axis=-1)` over a C-order array whose last axis has `row_len` elements: each
/// contiguous row reduces like [`sum`].
pub fn sum_last_axis(data: &[f64], row_len: usize) -> Vec<f64> {
    if row_len == 0 {
        return Vec::new();
    }
    assert_eq!(
        data.len() % row_len,
        0,
        "data is not a whole number of rows"
    );
    data.chunks_exact(row_len).map(sum).collect()
}

/// `np.mean(m, axis=-1)`: each row's [`mean`].
pub fn mean_last_axis(data: &[f64], row_len: usize) -> Vec<f64> {
    if row_len == 0 {
        return Vec::new();
    }
    assert_eq!(
        data.len() % row_len,
        0,
        "data is not a whole number of rows"
    );
    data.chunks_exact(row_len).map(mean).collect()
}

/// `np.nanmean(a)` over `float64`: NaNs become `0.0`, the array reduces like [`sum`], and
/// the total is divided by the count of non-NaN values. All-NaN or empty input gives NaN.
pub fn nanmean(a: &[f64]) -> f64 {
    let replaced: Vec<f64> = a
        .iter()
        .map(|&x| if x.is_nan() { 0.0 } else { x })
        .collect();
    let count = a.iter().filter(|x| !x.is_nan()).count();
    sum(&replaced) / count as f64
}

/// `np.cumsum(a)`: the first element as is, then each running total in order.
pub fn cumsum(a: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len());
    let mut acc = 0.0;
    for (i, &x) in a.iter().enumerate() {
        acc = if i == 0 { x } else { acc + x };
        out.push(acc);
    }
    out
}

/// `np.average(a, weights=w)` over 1-D arrays: `sum(a * w) / sum(w)`, each sum reduced
/// like [`sum`]. Weights that sum to zero are an error, as in numpy.
pub fn average(a: &[f64], w: &[f64]) -> Result<f64, NumericError> {
    if a.len() != w.len() {
        return Err(NumericError::ShapeMismatch);
    }
    let scale = sum(w);
    if scale == 0.0 {
        return Err(NumericError::WeightsSumToZero);
    }
    let products: Vec<f64> = a.iter().zip(w).map(|(&x, &y)| x * y).collect();
    Ok(sum(&products) / scale)
}

/// Python's builtin `sum(values)` over `np.float64` values, and
/// `functools.reduce(operator.add, values, 0)` (`moteval.metrics.base._plain_sum`): in
/// order, from `0.0`. The integer start `0` turns a leading `-0.0` into `0.0`, as `0.0`
/// does here.
pub fn plain_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut acc = 0.0;
    for x in values {
        acc += x;
    }
    acc
}

/// `np.arange(start, stop, step)` with float arguments.
///
/// numpy's length is `ceil((stop - start) / step)`. It stores `start`, then
/// `start + step`, and fills element `i >= 2` with `start + i * delta`, where
/// `delta = (start + step) - start` (`@NAME@_fill` in numpy's `arraytypes.c.src`).
pub fn arange(start: f64, stop: f64, step: f64) -> Result<Vec<f64>, NumericError> {
    if step == 0.0 {
        return Err(NumericError::ArangeZeroStep);
    }
    let span = stop - start;
    let value = span / step;
    // `_calc_length`: an underflowing quotient of a nonzero span has length 0 or 1.
    let len: i64 = if value == 0.0 && span != 0.0 {
        if value.is_sign_negative() { 0 } else { 1 }
    } else {
        let ceiled = value.ceil();
        if ceiled.is_nan() {
            return Err(NumericError::ArangeLength);
        }
        if !((i64::MIN as f64) <= ceiled && ceiled <= (i64::MAX as f64)) {
            return Err(NumericError::ArangeSize);
        }
        ceiled as i64
    };
    if len <= 0 {
        return Ok(Vec::new());
    }
    let len = usize::try_from(len).map_err(|_| NumericError::ArangeSize)?;
    let mut out = Vec::with_capacity(len);
    out.push(start);
    if len == 1 {
        return Ok(out);
    }
    let next = start + step;
    out.push(next);
    let delta = next - start;
    for i in 2..len {
        out.push(start + i as f64 * delta);
    }
    Ok(out)
}

/// `np.linspace(start, stop, num, endpoint)` for `float64` (numpy 2.5.1 `function_base.py`).
///
/// With `div = num - 1` (or `num` without the endpoint) and `step = (stop - start) / div`,
/// element `i` is `i * step + start`. When `step` is zero it is `i / div * delta + start`,
/// and when `div` is zero it is `i * delta + start`. With the endpoint and `num > 1`, the
/// last element is `stop` itself.
pub fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> Vec<f64> {
    let div = if endpoint { num.saturating_sub(1) } else { num };
    let delta = stop - start;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64).collect();
    if div > 0 {
        let div = div as f64;
        let step = delta / div;
        if step == 0.0 {
            for v in &mut y {
                *v /= div;
                *v *= delta;
            }
        } else {
            for v in &mut y {
                *v *= step;
            }
        }
    } else {
        for v in &mut y {
            *v *= delta;
        }
    }
    for v in &mut y {
        *v += start;
    }
    if endpoint && num > 1 {
        y[num - 1] = stop;
    }
    y
}

/// `np.round(x)` (zero decimals) and `np.rint(x)`: round half to even.
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// `np.maximum(a, b)`: NaN if either is NaN; otherwise the larger, and `b` on a tie, so
/// `maximum(0.0, -0.0)` is `-0.0`.
pub fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || a > b { a } else { b }
}

/// `np.minimum(a, b)`: NaN if either is NaN; otherwise the smaller, and `b` on a tie, so
/// `minimum(0.0, -0.0)` is `-0.0`.
pub fn minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || a < b { a } else { b }
}

/// Python's builtin `max(a, b)`: `b` only when `b > a`, so `max(1.0, nan)` is `1.0` and
/// `max(0.0, -0.0)` is `0.0`. Not `np.maximum`.
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's builtin `min(a, b)`: `b` only when `b < a`. Not `np.minimum`.
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_zero_and_nan_rules() {
        assert_eq!(maximum(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(maximum(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(minimum(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert!(maximum(f64::NAN, 1.0).is_nan());
        assert!(maximum(1.0, f64::NAN).is_nan());
        assert_eq!(py_max(1.0, f64::NAN), 1.0);
        assert!(py_max(f64::NAN, 1.0).is_nan());
        assert_eq!(py_max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(sum(&[-0.0, -0.0]).to_bits(), 0.0f64.to_bits());
        assert_eq!(pairwise_sum(&[-0.0, -0.0]).to_bits(), (-0.0f64).to_bits());
        assert_eq!(plain_sum([-0.0]).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn round_is_half_to_even() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(-0.5).to_bits(), (-0.0f64).to_bits());
    }
}
