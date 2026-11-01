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
//! | `np.log10(x)` for finite `x > 0` | [`log10`] |

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

/// `np.log10(x)` for a finite `x > 0`, as numpy 2.5.1 computes it on this host.
///
/// numpy's Linux x86-64 wheels compute float64 `log10` with Intel SVML
/// (`__svml_log108_ha`) when the CPU has the `X86_V4` features (AVX-512 F, CD, BW, DQ,
/// and VL), and with the platform libm otherwise (`loops_umath_fp.dispatch.c.src`). The
/// two differ in the last bit for about 1.4% of integers, so this function makes the same
/// choice at run time: [`log10_svml`] where numpy uses SVML, `f64::log10` (the same libm
/// numpy calls) everywhere else. numpy's `NPY_DISABLE_CPU_FEATURES` setting can move
/// numpy off SVML; this function does not read it.
pub fn log10(x: f64) -> f64 {
    if numpy_uses_svml() {
        log10_svml(x)
    } else {
        x.log10()
    }
}

/// Whether numpy 2.5.1 computes float64 `log10` with SVML on this host.
pub fn numpy_uses_svml() -> bool {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512cd")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        false
    }
}

const F64_FRACTION: u64 = (1 << 52) - 1;
const F64_ONE: u64 = 0x3FF0_0000_0000_0000;

/// numpy 2.5.1's `__svml_log108_ha` (`svml_z0_log10_d_ha.s`) for one finite `x > 0`, in
/// plain scalar code on any host. Each vector instruction maps to one operation here:
/// every `{rn-sae}` FMA is one `mul_add` (one rounding), and the constants are the bits
/// of `__svml_dlog10_ha_data_internal_avx512`. Zero, negative, infinite, and NaN inputs
/// take SVML's separate special-case path, which is not ported.
///
/// The kernel computes `log10(x) = k*log10(2) - log10(rcp) + log10(1 + R)`: `k` and the
/// mantissa `m` in `[1, 2)` from `vgetexppd`/`vgetmantpd`, `rcp` = `vrcp14pd(m)` rounded
/// to a multiple of 1/32 (`vrndscalepd $0x58`), and `R = rcp*m - 1`.
pub fn log10_svml(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite(), "log10_svml takes a finite x > 0");
    // vgetmantpd $8 and vgetexppd: a subnormal is first scaled into the normal range.
    let (bits, scale) = if x.to_bits() >> 52 == 0 {
        ((x * f64::from_bits(0x4350_0000_0000_0000)).to_bits(), -54) // 2**54
    } else {
        (x.to_bits(), 0)
    };
    let mantissa = f64::from_bits((bits & F64_FRACTION) | F64_ONE);
    let exponent = ((bits >> 52) as i64 - 1023 + scale) as f64;

    let rcp = (rcp14(mantissa) * 32.0).round_ties_even() / 32.0;
    let r = rcp.mul_add(mantissa, -1.0);
    // Add 1 to the exponent when rcp < 0.75; the table index is rcp's top 4 fraction bits.
    let k = if rcp < 0.75 { exponent + 1.0 } else { exponent };
    let index = ((rcp.to_bits() >> 48) & 15) as usize;
    let table_hi = f64::from_bits(SVML_LOG10_TABLE_HI[index]);
    let table_lo = f64::from_bits(SVML_LOG10_TABLE_LO[index]);
    let c = f64::from_bits;
    let log10_2_hi = c(0x3fd3_4413_509f_0000);
    let log10_2_lo = c(0x3d7e_7fbc_c47c_4acd);
    let c0_hi = c(0x3fdb_cb7b_1526_e50e); // 1/ln(10), high part
    let c0_lo = c(0x3c69_a5fd_aaf1_1825);

    let p1 = c(0x3faf_c3f6_f38b_58a2).mul_add(r, c(0xbfb2_87a6_3464_dc80));
    let p0 = c(0x3fa8_c2d8_2848_0370).mul_add(r, c(0xbfab_d80d_9602_9814));
    let r2 = r * r;
    let r4 = r2 * r2;
    let q = r2.mul_add(p0, p1);
    let k_hi = log10_2_hi.mul_add(k, table_hi);
    let k_lo = log10_2_lo.mul_add(k, table_lo);
    let s7 = c(0x3fc2_87a7_636f_428c).mul_add(r, c(0xbfcb_cb7b_1526_e4db));
    let s9 = c(0x3fb6_3c62_777f_27d9).mul_add(r, c(0xbfbb_cb7b_153c_06a3));
    let head = r.mul_add(c0_hi, k_hi);
    let s9 = r2.mul_add(s9, s7);
    let c0r_hi = head - k_hi;
    let poly = r4.mul_add(q, s9);
    let poly = r.mul_add(poly, c0_lo);
    let c0r_lo = r.mul_add(c0_hi, -c0r_hi);
    let tail = k_lo + c0r_lo;
    let poly = r.mul_add(poly, tail);
    head + poly
}

/// `-log10(rcp)` for each rounded reciprocal, high part, by permute index (SVML table
/// offsets 0 and 64).
const SVML_LOG10_TABLE_HI: [u64; 16] = [
    0x0000_0000_0000_0000,
    0xbf9a_f5f9_2b00_0000,
    0xbfaa_30a9_d60a_0000,
    0xbfb3_1b30_55c4_0000,
    0xbfb8_cf18_3886_0000,
    0xbfbe_3bc1_ab0e_0000,
    0xbfc1_b3e7_1ec9_8000,
    0xbfc4_2c7e_7fe4_0000,
    0x3fbf_fbfc_2bbc_0000,
    0x3fbb_721c_d171_0000,
    0x3fb7_15d0_ce36_0000,
    0x3fb2_e3a7_40b8_0000,
    0x3fad_b11e_d766_0000,
    0x3fa5_e396_6b7e_0000,
    0x3f9c_b38f_ccd8_0000,
    0x3f8c_3d08_3778_0000,
];

/// The low parts of [`SVML_LOG10_TABLE_HI`] (SVML table offsets 128 and 192).
const SVML_LOG10_TABLE_LO: [u64; 16] = [
    0x0000_0000_0000_0000,
    0xbd4c_c1f4_dbc1_4db5,
    0x3d20_163d_7e67_d282,
    0xbd5c_4600_6d08_2e6d,
    0xbd52_0326_ca2c_7e5f,
    0xbd39_fe3d_562a_53f0,
    0x3d58_42a2_2266_6db0,
    0x3d1f_f1d2_2abd_8abe,
    0x3d5e_00dd_60df_12c3,
    0x3d55_f8a5_7d98_2e77,
    0x3d5e_beeb_67bf_155a,
    0xbd5f_fc5a_ef53_f2ed,
    0x3d55_7e86_5b87_8377,
    0x3d52_52a3_7739_a5f7,
    0x3d47_fb6d_2d65_28c7,
    0x3d23_1027_2fe1_7537,
];

/// `vrcp14pd` for `m` in `[1, 2)`: `RCP14D` from Intel's reference implementation
/// (`RECIP14.c`, Intel Corporation 2015, BSD-3-Clause), which the instruction reference
/// calls numerically exact. The result depends only on the top 16 fraction bits of `m`.
/// On the AMD Zen 5 host that froze the reference, this equals the hardware instruction
/// for every top-16-bit input.
fn rcp14(m: f64) -> f64 {
    const CLEAR_LOW_36_BITS: u64 = 0xFFFF_FFF0_0000_0000;
    const EXPONENT_DELTA_18: u64 = 0x0120_0000_0000_0000; // divides by 2**18
    let bits = m.to_bits();
    if bits & F64_FRACTION == 0 {
        return 1.0; // an exact power of two
    }
    let i = ((m - 1.0) * 64.0).floor() as usize;
    let y = 1.0 + 1.0 / 128.0 + i as f64 / 64.0;
    let x_minus_y = f64::from_bits(bits & CLEAR_LOW_36_BITS) - y; // exact
    let a = f64::from(RCP14_COEFF[2 * i + 1]);
    let b = 256.0 * f64::from(RCP14_COEFF[2 * i]);
    let rcp = a - b * x_minus_y; // exact
    f64::from_bits((rcp.to_bits() & CLEAR_LOW_36_BITS) - EXPONENT_DELTA_18)
}

/// `RCP14_Coeff` from `RECIP14.c`: `b[i]` at `2*i`, `a[i]` at `2*i + 1`.
#[rustfmt::skip]
const RCP14_COEFF: [u32; 128] = [
    1009, 260119, 977, 256148, 949, 252296, 921, 248558,
    893, 244929, 869, 241405, 843, 237981, 821, 234652,
    797, 231416, 777, 228266, 755, 225202, 735, 222220,
    717, 219314, 699, 216485, 681, 213727, 663, 211038,
    647, 208417, 631, 205859, 617, 203364, 601, 200929,
    587, 198551, 573, 196229, 561, 193960, 547, 191743,
    535, 189576, 523, 187458, 513, 185387, 501, 183360,
    491, 181377, 479, 179439, 469, 177540, 459, 175681,
    451, 173860, 441, 172077, 433, 170330, 423, 168618,
    415, 166940, 407, 165295, 399, 163682, 391, 162101,
    385, 160550, 377, 159027, 369, 157535, 363, 156069,
    357, 154631, 349, 153219, 343, 151832, 337, 150470,
    331, 149133, 325, 147819, 319, 146528, 315, 145260,
    309, 144012, 303, 142787, 299, 141582, 293, 140397,
    289, 139232, 285, 138085, 279, 136959, 275, 135853,
    271, 134763, 267, 133689, 263, 132631, 259, 131589,
];

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
