use crate::model::{BoxXywh, Matrix};
use crate::numeric::{EPS, maximum, minimum};
use crate::rle::{self, RleError, RleMask};

struct Corners {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    // TrackEval's corner-derived area, not `w * h`: they differ by an ulp when `x + w` rounds.
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

fn intersection(a: &Corners, b: &Corners) -> f64 {
    let ix1 = maximum(a.x1, b.x1);
    let iy1 = maximum(a.y1, b.y1);
    let ix2 = minimum(a.x2, b.x2);
    let iy2 = minimum(a.y2, b.y2);
    maximum(ix2 - ix1, 0.0) * maximum(iy2 - iy1, 0.0)
}

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

pub fn mask_iou(masks_a: &[RleMask], masks_b: &[RleMask]) -> Result<Matrix, RleError> {
    let ious = rle::iou(masks_a, masks_b, &vec![false; masks_b.len()])?;
    Ok(matrix(masks_a.len(), masks_b.len(), ious))
}

pub fn mask_ioa(masks_a: &[RleMask], masks_b: &[RleMask]) -> Result<Matrix, RleError> {
    let ioas = rle::iou(masks_a, masks_b, &vec![true; masks_b.len()])?;
    Ok(matrix(masks_a.len(), masks_b.len(), ioas))
}

fn matrix(n: usize, m: usize, values: Vec<f64>) -> Matrix {
    if n == 0 || m == 0 {
        return Matrix::zeros(n, m);
    }
    Matrix::new(n, m, values).expect("pycocotools iou returns n * m values")
}

pub fn merge_masks(rles: &[RleMask]) -> Result<RleMask, RleError> {
    rle::merge(rles, false)
}

pub fn masks_overlap(rles: &[RleMask]) -> Result<bool, RleError> {
    let merged_area = rle::area_u32(&rle::merge(rles, false)?);
    let total_area = rles
        .iter()
        .fold(0u32, |total, mask| total.wrapping_add(rle::area_u32(mask)));
    Ok(total_area > merged_area)
}
