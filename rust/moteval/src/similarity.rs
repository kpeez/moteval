//! Geometry similarity, ported from `moteval/data/similarity.py`.
//!
//! Boxes: Jaccard IoU and intersection-over-area on `xywh` rows, element for element as
//! numpy computes them (TrackEval's `_calculate_box_ious`). Masks: pycocotools RLE IoU and
//! IoA through `moteval::rle`, as TrackEval's `_calculate_mask_ious` calls them.

use crate::model::{BoxXywh, Matrix};
use crate::numeric::{EPS, maximum, minimum};
use crate::rle::{self, RleError, RleMask};

/// Corners and corner-derived area of one `xywh` box, as `box_iou` derives them.
struct Corners {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    /// `(x2 - x1) * (y2 - y1)`: differs from `w * h` by an ulp when `x + w` rounds, and
    /// LocA's similarity sums see that difference.
    area: f64,
}

fn corners(b: &BoxXywh) -> Corners {
    let [x1, y1, w, h] = *b;
    let x2 = x1 + w;
    let y2 = y1 + h;
    Corners {
        x1,
        y1,
        x2,
        y2,
        area: (x2 - x1) * (y2 - y1),
    }
}

/// `np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)` for one pair.
fn intersection(a: &Corners, b: &Corners) -> f64 {
    let ix1 = maximum(a.x1, b.x1);
    let iy1 = maximum(a.y1, b.y1);
    let ix2 = minimum(a.x2, b.x2);
    let iy2 = minimum(a.y2, b.y2);
    maximum(ix2 - ix1, 0.0) * maximum(iy2 - iy1, 0.0)
}

/// `box_iou(boxes_a, boxes_b)`: the `(n, m)` IoU matrix between two sets of `xywh` boxes.
///
/// The intersection is zeroed where either area or the union is at most `EPS`; such a
/// union becomes `1.0`, so the result there is `0.0`.
pub fn box_iou(boxes_a: &[BoxXywh], boxes_b: &[BoxXywh]) -> Matrix {
    let (n, m) = (boxes_a.len(), boxes_b.len());
    if n == 0 || m == 0 {
        return Matrix::zeros(n, m);
    }
    let a: Vec<Corners> = boxes_a.iter().map(corners).collect();
    let b: Vec<Corners> = boxes_b.iter().map(corners).collect();
    let mut out = Vec::with_capacity(n * m);
    for ca in &a {
        for cb in &b {
            let mut inter = intersection(ca, cb);
            let mut union = ca.area + cb.area - inter;
            // numpy zeroes `inter` by rows, then columns, then by the original union, and
            // only then replaces small unions; each test reads the unmodified union.
            if ca.area <= EPS || cb.area <= EPS || union <= EPS {
                inter = 0.0;
            }
            if union <= EPS {
                union = 1.0;
            }
            out.push(inter / union);
        }
    }
    Matrix::new(n, m, out).expect("n * m values")
}

/// `box_ioa(boxes_a, boxes_b)`: the `(n, m)` intersection-over-area matrix, normalised by
/// each `boxes_a` box's corner-derived area; rows whose area is at most `EPS` are `0.0`.
pub fn box_ioa(boxes_a: &[BoxXywh], boxes_b: &[BoxXywh]) -> Matrix {
    let (n, m) = (boxes_a.len(), boxes_b.len());
    if n == 0 || m == 0 {
        return Matrix::zeros(n, m);
    }
    let a: Vec<Corners> = boxes_a.iter().map(corners).collect();
    let b: Vec<Corners> = boxes_b.iter().map(corners).collect();
    let mut out = Vec::with_capacity(n * m);
    for ca in &a {
        let valid = ca.area > EPS;
        for cb in &b {
            out.push(if valid {
                intersection(ca, cb) / ca.area
            } else {
                0.0
            });
        }
    }
    Matrix::new(n, m, out).expect("n * m values")
}

/// `mask_iou(masks_a, masks_b)`: `pycocotools.mask.iou` with every `iscrowd` false, as a
/// `(len(a), len(b))` matrix. Pairs whose boxes do not overlap are `0.0`; a compared pair
/// of different sizes is `-1.0`, as pycocotools gives.
pub fn mask_iou(masks_a: &[RleMask], masks_b: &[RleMask]) -> Result<Matrix, RleError> {
    let ious = rle::iou(masks_a, masks_b, &vec![false; masks_b.len()])?;
    Ok(matrix(masks_a.len(), masks_b.len(), ious))
}

/// `mask_ioa(masks_a, masks_b)`: `pycocotools.mask.iou` with every `iscrowd` true, so the
/// denominator is each `masks_a` mask's own area.
pub fn mask_ioa(masks_a: &[RleMask], masks_b: &[RleMask]) -> Result<Matrix, RleError> {
    let ioas = rle::iou(masks_a, masks_b, &vec![true; masks_b.len()])?;
    Ok(matrix(masks_a.len(), masks_b.len(), ioas))
}

/// A `pycocotools.mask.iou` result as a matrix; empty input keeps its `(n, 0)` or `(0, m)`
/// shape, as `similarity.py` reshapes it.
fn matrix(n: usize, m: usize, values: Vec<f64>) -> Matrix {
    if n == 0 || m == 0 {
        return Matrix::zeros(n, m);
    }
    Matrix::new(n, m, values).expect("pycocotools iou returns n * m values")
}

/// `merge_masks(rles)`: the union of the masks (`mask.merge(intersect=False)`). No masks
/// give the `0 x 0` empty mask, and so do masks of different sizes.
pub fn merge_masks(rles: &[RleMask]) -> Result<RleMask, RleError> {
    rle::merge(rles, false)
}

/// `masks_overlap(rles)`: whether any two masks in the list overlap.
///
/// Python compares `sum(area(rle) for rle in rles)` with the merged mask's area. Both are
/// `np.uint32`, so the sum wraps at `2**32` as it does here.
pub fn masks_overlap(rles: &[RleMask]) -> Result<bool, RleError> {
    let merged_area = rle::area_u32(&rle::merge(rles, false)?);
    let total_area = rles
        .iter()
        .fold(0u32, |total, mask| total.wrapping_add(rle::area_u32(mask)));
    Ok(total_area > merged_area)
}
