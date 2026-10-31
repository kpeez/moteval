//! Geometry similarity, ported from `moteval/data/similarity.py`.
//!
//! Boxes: Jaccard IoU and intersection-over-area on `xywh` rows, element for element as
//! numpy computes them (TrackEval's `_calculate_box_ious`). Masks: pycocotools RLE IoU and
//! IoA (the rle-masks-jf package fills those bodies).

use crate::model::{BoxXywh, Matrix};
use crate::numeric::{EPS, maximum, minimum};
use crate::rle::{RleError, RleMask};

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
/// `(len(a), len(b))` matrix.
pub fn mask_iou(_masks_a: &[RleMask], _masks_b: &[RleMask]) -> Result<Matrix, RleError> {
    todo!("rle-masks-jf: port moteval.data.similarity.mask_iou")
}

/// `mask_ioa(masks_a, masks_b)`: `pycocotools.mask.iou` with every `iscrowd` true, so the
/// denominator is each `masks_a` mask's own area.
pub fn mask_ioa(_masks_a: &[RleMask], _masks_b: &[RleMask]) -> Result<Matrix, RleError> {
    todo!("rle-masks-jf: port moteval.data.similarity.mask_ioa")
}

/// `merge_masks(rles)`: the union of the masks (`mask.merge(intersect=False)`).
pub fn merge_masks(_rles: &[RleMask]) -> Result<RleMask, RleError> {
    todo!("rle-masks-jf: port moteval.data.similarity.merge_masks")
}

/// `masks_overlap(rles)`: whether any two masks in the list overlap.
pub fn masks_overlap(_rles: &[RleMask]) -> Result<bool, RleError> {
    todo!("rle-masks-jf: port moteval.data.similarity.masks_overlap")
}
