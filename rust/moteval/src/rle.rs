//! pycocotools-compatible run-length encoded masks.
//!
//! A port of pycocotools 2.0.11 `common/maskApi.c` and the parts of `_mask.pyx` moteval
//! calls: `encode` (Fortran order, one `h x w` mask), `decode`, `area`, `to_bbox`,
//! `merge` (union and intersect), `iou` (with `iscrowd`), and the compressed counts string
//! (`rleFrString` / `rleToString`).
//!
//! Python holds a mask as `{"size": [h, w], "counts": bytes}` and every pycocotools call
//! parses the string again with `rleFrString`. `RleMask` keeps the bytes as given and the
//! counts `rleFrString` parses from them, so each operation sees what the C code sees.
//! Masks built here (`encode`, `merge`) hold the string `rleToString` writes, parsed back,
//! as Python's dict does.
//!
//! C `uint` arithmetic is `u32` with wrapping, as in the C code. Two behaviours differ from
//! pycocotools on purpose, both on malformed input only: a counts string whose runs do
//! not cover exactly `h * w` pixels is an error at construction (pycocotools raises on an
//! overrun in `decode` and reads uninitialised memory on a shortfall), and a string that
//! ends inside a continuation group ends there (the C parser reads past the terminator).

use std::borrow::Borrow;
use std::fmt;

/// One mask in pycocotools RLE form: `{"size": [height, width], "counts": bytes}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RleMask {
    height: usize,
    width: usize,
    /// The compressed counts string, exactly as given (Python's `counts` bytes).
    counts: Vec<u8>,
    /// `rleFrString(counts)`: the run lengths, background first.
    cnts: Vec<u32>,
}

impl RleMask {
    /// A mask from its compressed counts string (the `counts` bytes pycocotools stores).
    ///
    /// The string is parsed as `rleFrString` parses it, up to the first NUL byte. Runs that
    /// do not add up to `height * width` pixels are `RleError::InvalidCounts`.
    pub fn from_compressed(height: usize, width: usize, counts: &[u8]) -> Result<Self, RleError> {
        let cnts = fr_string(counts);
        let total: u128 = cnts.iter().map(|&c| u128::from(c)).sum();
        if total != height as u128 * width as u128 {
            return Err(RleError::InvalidCounts);
        }
        Ok(Self {
            height,
            width,
            counts: counts.to_vec(),
            cnts,
        })
    }

    /// The mask pycocotools returns for run lengths `cnts`: `rleToString` writes the string
    /// and the counts are parsed back from it, as Python's next call would.
    fn from_cnts(height: usize, width: usize, cnts: &[u32]) -> Self {
        let counts = to_string(cnts);
        let cnts = fr_string(&counts);
        Self {
            height,
            width,
            counts,
            cnts,
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// The compressed counts string, as pycocotools writes it.
    pub fn to_compressed(&self) -> Vec<u8> {
        self.counts.clone()
    }

    /// The run lengths, background first (`rleFrString` of the counts string).
    pub fn runs(&self) -> &[u32] {
        &self.cnts
    }
}

/// Errors from RLE decoding and mask operations. Messages match the Python text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RleError {
    /// A counts string whose runs do not cover `height * width` pixels. pycocotools'
    /// `decode` raises `ValueError` with this text when the runs overrun the mask.
    InvalidCounts,
    /// `iou` with an `iscrowd` list whose length is not the number of `gt` masks
    /// (pycocotools: `AssertionError`).
    CrowdLength,
}

impl fmt::Display for RleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCounts => f.write_str("Invalid RLE mask representation"),
            Self::CrowdLength => f.write_str("iou(iscrowd=) must have the same length as gt"),
        }
    }
}

impl std::error::Error for RleError {}

/// `rleToString`: LEB128-like, 6 bits per character, ASCII 48 to 111. From the fourth run
/// on, each run is stored as the difference to the run two before it.
fn to_string(cnts: &[u32]) -> Vec<u8> {
    let mut s = Vec::with_capacity(cnts.len() * 2);
    for (i, &cnt) in cnts.iter().enumerate() {
        let mut x = i64::from(cnt);
        if i > 2 {
            x -= i64::from(cnts[i - 2]);
        }
        let mut more = true;
        while more {
            let mut c = (x & 0x1f) as u8;
            x >>= 5;
            more = if c & 0x10 != 0 { x != -1 } else { x != 0 };
            if more {
                c |= 0x20;
            }
            s.push(c + 48);
        }
    }
    s
}

/// `rleFrString`, up to the first NUL byte.
///
/// The C code shifts in `int`: each group is `(c & 0x1f) << 5*k` and the sign extension is
/// `-1 << 5*k`, both 32-bit and sign-extended into the `long` accumulator. x86's shift
/// count is taken mod 32; `wrapping_shl` does the same. Well-formed strings never reach a
/// shift above 30.
fn fr_string(s: &[u8]) -> Vec<u32> {
    let end = s.iter().position(|&b| b == 0).unwrap_or(s.len());
    let s = &s[..end];
    let mut cnts: Vec<u32> = Vec::with_capacity(s.len());
    let mut p = 0;
    while p < s.len() {
        let mut x: i64 = 0;
        let mut k: u32 = 0;
        let mut more = true;
        while more {
            let Some(&byte) = s.get(p) else {
                // The C loop would read past the terminator; stop at the end instead.
                break;
            };
            let c = i32::from((byte as i8).wrapping_sub(48));
            x |= i64::from((c & 0x1f).wrapping_shl(5 * k));
            more = c & 0x20 != 0;
            p += 1;
            k += 1;
            if !more && (c & 0x10) != 0 {
                x |= i64::from((-1i32).wrapping_shl(5 * k));
            }
        }
        let m = cnts.len();
        if m > 2 {
            x += i64::from(cnts[m - 2]);
        }
        cnts.push(x as u32);
    }
    cnts
}

/// `pycocotools.mask.encode` of one Fortran-order `height x width` mask (`rleEncode`).
///
/// A run ends wherever the byte value changes, as in the C code; moteval passes `0`/`1`
/// masks only.
///
/// # Panics
///
/// If `mask_fortran` does not hold `height * width` bytes.
pub fn encode(mask_fortran: &[u8], height: usize, width: usize) -> RleMask {
    assert_eq!(
        mask_fortran.len(),
        height * width,
        "mask is not height x width"
    );
    let mut cnts = Vec::new();
    let mut p = 0u8;
    let mut c = 0u32;
    for &v in mask_fortran {
        if v != p {
            cnts.push(c);
            c = 0;
            p = v;
        }
        c = c.wrapping_add(1);
    }
    cnts.push(c);
    RleMask::from_cnts(height, width, &cnts)
}

/// `pycocotools.mask.decode`: the binary mask in Fortran order (`rleDecode`).
pub fn decode(rle: &RleMask) -> Vec<u8> {
    let mut out = Vec::with_capacity(rle.height * rle.width);
    let mut v = 0u8;
    for &cnt in &rle.cnts {
        out.extend(std::iter::repeat_n(v, cnt as usize));
        v ^= 1;
    }
    out
}

/// `pycocotools.mask.area`: the number of foreground pixels (`rleArea`, a C `uint` sum).
pub fn area(rle: &RleMask) -> u64 {
    u64::from(area_u32(rle))
}

/// `rleArea` as C computes it: the odd runs, summed in `uint`.
pub(crate) fn area_u32(rle: &RleMask) -> u32 {
    rle.cnts
        .iter()
        .skip(1)
        .step_by(2)
        .fold(0u32, |a, &c| a.wrapping_add(c))
}

/// `pycocotools.mask.toBbox`: `[x, y, w, h]` (`rleToBbox`).
///
/// A foreground run that spans columns widens the box to the full height. An odd number
/// of runs drops the last (background) run. Where the C code would divide by a zero
/// height (a nonzero foreground run in a 0-pixel mask, which construction rejects), this
/// returns zeros.
pub fn to_bbox(rle: &RleMask) -> [f64; 4] {
    let h = rle.height as u32;
    let w = rle.width as u32;
    let m = (rle.cnts.len() / 2) * 2;
    if m == 0 {
        return [0.0; 4];
    }
    let (mut xs, mut ys, mut xe, mut ye) = (w, h, 0u32, 0u32);
    let mut cc = 0u32;
    for j in 0..m {
        let start = cc;
        cc = cc.wrapping_add(rle.cnts[j]);
        if j % 2 == 0 || rle.cnts[j] == 0 {
            continue;
        }
        if h == 0 {
            return [0.0; 4];
        }
        let y_start = start % h;
        let x_start = (start - y_start) / h;
        let y_end = cc.wrapping_sub(1) % h;
        let x_end = cc.wrapping_sub(1).wrapping_sub(y_end) / h;
        xs = xs.min(x_start);
        xe = xe.max(x_end);
        if x_start < x_end {
            ys = 0;
            ye = h - 1;
        } else {
            ys = ys.min(y_start);
            ye = ye.max(y_end);
        }
    }
    [
        f64::from(xs),
        f64::from(ys),
        f64::from(xe.wrapping_sub(xs).wrapping_add(1)),
        f64::from(ye.wrapping_sub(ys).wrapping_add(1)),
    ]
}

/// `pycocotools.mask.merge(rles, intersect)` (`rleMerge`).
///
/// No masks give a `0 x 0` mask with no runs; one mask is returned re-encoded. Masks of
/// different sizes give the `0 x 0` empty mask too, as pycocotools does; it never raises.
pub fn merge(rles: &[RleMask], intersect: bool) -> Result<RleMask, RleError> {
    let Some(first) = rles.first() else {
        return Ok(RleMask::from_cnts(0, 0, &[]));
    };
    let (mut h, mut w) = (first.height, first.width);
    if rles.len() == 1 {
        return Ok(RleMask::from_cnts(h, w, &first.cnts));
    }
    let mut cnts: Vec<u32> = first.cnts.clone();
    for b in &rles[1..] {
        if b.height != h || b.width != w {
            h = 0;
            w = 0;
            cnts.clear();
            break;
        }
        let a_cnts = std::mem::take(&mut cnts);
        let run = |runs: &[u32], i: usize| runs.get(i).copied().unwrap_or(0);
        let (mut ca, mut cb) = (run(&a_cnts, 0), run(&b.cnts, 0));
        let (mut v, mut va, mut vb) = (false, false, false);
        let (mut a, mut bi) = (1usize, 1usize);
        let mut cc = 0u32;
        let mut ct = 1u32;
        while ct > 0 {
            let c = ca.min(cb);
            cc = cc.wrapping_add(c);
            ct = 0;
            ca -= c;
            if ca == 0 && a < a_cnts.len() {
                ca = a_cnts[a];
                a += 1;
                va = !va;
            }
            ct = ct.wrapping_add(ca);
            cb -= c;
            if cb == 0 && bi < b.cnts.len() {
                cb = b.cnts[bi];
                bi += 1;
                vb = !vb;
            }
            ct = ct.wrapping_add(cb);
            let vp = v;
            v = if intersect { va && vb } else { va || vb };
            if v != vp || ct == 0 {
                cnts.push(cc);
                cc = 0;
            }
        }
    }
    Ok(RleMask::from_cnts(h, w, &cnts))
}

/// `bbIou` for one pair: `0.0` unless the boxes overlap with positive width and height.
fn bb_iou(d: &[f64; 4], g: &[f64; 4], crowd: bool) -> f64 {
    let ga = g[2] * g[3];
    let da = d[2] * d[3];
    let w = (d[2] + d[0]).min(g[2] + g[0]) - d[0].max(g[0]);
    if w <= 0.0 {
        return 0.0;
    }
    let h = (d[3] + d[1]).min(g[3] + g[1]) - d[1].max(g[1]);
    if h <= 0.0 {
        return 0.0;
    }
    let i = w * h;
    let u = if crowd { da } else { da + ga - i };
    i / u
}

/// `pycocotools.mask.iou(dt, gt, iscrowd)` (`rleIou`): a `dt.len() x gt.len()` matrix in C
/// order.
///
/// Only pairs whose bounding boxes overlap are compared run by run; the rest are `0.0`.
/// A compared pair of different sizes is `-1.0`. With `iscrowd[g]` the denominator is the
/// `dt` mask's area. Empty `dt` or `gt` gives an empty result. Takes `&[RleMask]` or
/// `&[&RleMask]`.
pub fn iou<M: Borrow<RleMask>>(dt: &[M], gt: &[M], iscrowd: &[bool]) -> Result<Vec<f64>, RleError> {
    if iscrowd.len() != gt.len() {
        return Err(RleError::CrowdLength);
    }
    let (m, n) = (dt.len(), gt.len());
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let db: Vec<[f64; 4]> = dt.iter().map(|d| to_bbox(d.borrow())).collect();
    let gb: Vec<[f64; 4]> = gt.iter().map(|g| to_bbox(g.borrow())).collect();
    let mut out = vec![0.0; m * n];
    for (g, gt_mask) in gt.iter().map(Borrow::borrow).enumerate() {
        let crowd = iscrowd[g];
        for (d, dt_mask) in dt.iter().map(Borrow::borrow).enumerate() {
            // C: `if (o[g*m+d] > 0)`, so a NaN box IoU is skipped too.
            if bb_iou(&db[d], &gb[g], crowd) > 0.0 {
                out[d * n + g] =
                    if dt_mask.height != gt_mask.height || dt_mask.width != gt_mask.width {
                        -1.0
                    } else {
                        rle_pair_iou(dt_mask, gt_mask, crowd)
                    };
            }
        }
    }
    Ok(out)
}

/// The run-by-run body of `rleIou` for one same-size pair whose boxes overlap.
fn rle_pair_iou(dt: &RleMask, gt: &RleMask, crowd: bool) -> f64 {
    let run = |runs: &[u32], i: usize| runs.get(i).copied().unwrap_or(0);
    let (ka, kb) = (dt.cnts.len(), gt.cnts.len());
    let (mut ca, mut cb) = (run(&dt.cnts, 0), run(&gt.cnts, 0));
    let (mut va, mut vb) = (false, false);
    let (mut a, mut b) = (1usize, 1usize);
    let (mut i, mut u) = (0u32, 0u32);
    let mut ct = 1u32;
    while ct > 0 {
        let c = ca.min(cb);
        if va || vb {
            u = u.wrapping_add(c);
            if va && vb {
                i = i.wrapping_add(c);
            }
        }
        ct = 0;
        ca -= c;
        if ca == 0 && a < ka {
            ca = dt.cnts[a];
            a += 1;
            va = !va;
        }
        ct = ct.wrapping_add(ca);
        cb -= c;
        if cb == 0 && b < kb {
            cb = gt.cnts[b];
            b += 1;
            vb = !vb;
        }
        ct = ct.wrapping_add(cb);
    }
    if i == 0 {
        u = 1;
    } else if crowd {
        u = area_u32(dt);
    }
    f64::from(i) / f64::from(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_codec_round_trips_negative_deltas() {
        let cnts = [3, 40, 1, 2, 100_000, 0, 7, 1, 65_535];
        let s = to_string(&cnts);
        assert!(s.iter().all(|&c| (48..112).contains(&c)));
        assert_eq!(fr_string(&s), cnts);
    }

    #[test]
    fn non_canonical_strings_parse_like_c() {
        // 'Q' is 1 with the continuation bit; '0' then ends the group: x = 1.
        assert_eq!(fr_string(b"Q0"), [1]);
        // A NUL byte ends the string, as in C.
        assert_eq!(fr_string(b"1\x002"), [1]);
    }

    #[test]
    fn construction_rejects_runs_that_miss_the_pixel_count() {
        assert_eq!(
            RleMask::from_compressed(2, 2, b"3"),
            Err(RleError::InvalidCounts)
        );
        assert!(RleMask::from_compressed(2, 2, b"4").is_ok());
        assert!(RleMask::from_compressed(0, 0, b"").is_ok());
    }

    #[test]
    fn encode_decode_area_bbox() {
        // 3 x 2, Fortran order: column 0 = [0, 1, 1], column 1 = [0, 0, 1].
        let mask = [0, 1, 1, 0, 0, 1];
        let rle = encode(&mask, 3, 2);
        assert_eq!(rle.runs(), [1, 2, 2, 1]);
        assert_eq!(decode(&rle), mask);
        assert_eq!(area(&rle), 3);
        // The run from (1, 0) to (2, 0) stays in column 0; (2, 1) is in column 1.
        assert_eq!(to_bbox(&rle), [0.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn merge_of_nothing_or_mismatched_sizes_is_empty() {
        let empty = merge(&[], false).unwrap();
        assert_eq!(
            (empty.height(), empty.width(), empty.runs()),
            (0, 0, &[][..])
        );
        let a = encode(&[1, 0], 2, 1);
        let b = encode(&[1, 0, 0], 3, 1);
        let mixed = merge(&[a, b], false).unwrap();
        assert_eq!(
            (mixed.height(), mixed.width(), mixed.runs()),
            (0, 0, &[][..])
        );
    }
}
