//! The shared preprocessing engine, ported from `moteval/data/protocol.py`.
//!
//! `preprocess_frame` runs any `Protocol` on one frame, as TrackEval's
//! `get_preprocessed_seq_data` does (commit `12c8791b`): keep the class's predictions,
//! Hungarian-match them against all GT and drop those matched to a distractor class, drop
//! unmatched predictions inside an ignore region, then keep the class's GT (minus
//! conf-zero rows when declared). The engine reads precomputed similarity and ignore IoA,
//! so boxes and masks share it. No per-benchmark hooks.
//!
//! The `Protocol` type itself lives in `crate::model`.

use crate::lsap::{LsapError, linear_sum_assignment};
use crate::model::{Matrix, Protocol};
use crate::numeric::EPS;

/// One frame of raw GT and prediction arrays plus precomputed geometry. `D` is the
/// detection type: `BoxXywh` or `RleMask`.
///
/// `similarity` is `(num_gt, num_pred)`; `ignore_ioa` is `(num_pred, num_ignore)`. Ids
/// are raw (not densified).
#[derive(Clone, Debug)]
pub struct RawFrame<D> {
    pub gt_ids: Vec<i64>,
    pub gt_classes: Vec<i64>,
    pub gt_conf: Vec<f64>,
    pub gt_dets: Vec<D>,
    pub pred_ids: Vec<i64>,
    pub pred_classes: Vec<i64>,
    pub pred_confidences: Vec<f64>,
    pub pred_dets: Vec<D>,
    pub similarity: Matrix,
    pub ignore_ioa: Matrix,
}

/// One frame after preprocessing: only the surviving GT and predictions.
#[derive(Clone, Debug)]
pub struct PreprocessedFrame<D> {
    pub gt_ids: Vec<i64>,
    pub gt_dets: Vec<D>,
    pub pred_ids: Vec<i64>,
    pub pred_confidences: Vec<f64>,
    pub pred_dets: Vec<D>,
    pub similarity: Matrix,
}

/// `values[indices]` for a 1-D array.
fn take<T: Clone>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| values[i].clone()).collect()
}

/// The columns `cols` of `m`, in that order (`m[:, cols]`).
fn take_cols(m: &Matrix, cols: &[usize]) -> Matrix {
    let mut data = Vec::with_capacity(m.rows() * cols.len());
    for r in 0..m.rows() {
        data.extend(cols.iter().map(|&c| m.get(r, c)));
    }
    Matrix::new(m.rows(), cols.len(), data).expect("rows x cols values")
}

/// The rows `rows` of `m`, in that order (`m[rows]`).
fn take_rows(m: &Matrix, rows: &[usize]) -> Matrix {
    let mut data = Vec::with_capacity(rows.len() * m.cols());
    for &r in rows {
        data.extend_from_slice(m.row(r));
    }
    Matrix::new(rows.len(), m.cols(), data).expect("rows x cols values")
}

/// Run the preprocessing engine on one frame for class `cls_id`.
///
/// Fails only where scipy's `linear_sum_assignment` raises: a NaN similarity reaches the
/// matching step.
pub fn preprocess_frame<D: Clone>(
    frame: &RawFrame<D>,
    protocol: &Protocol,
    cls_id: i64,
) -> Result<PreprocessedFrame<D>, LsapError> {
    // Step 1: keep only predictions of the class under evaluation.
    let pred_keep: Vec<usize> = (0..frame.pred_classes.len())
        .filter(|&j| frame.pred_classes[j] == cls_id)
        .collect();
    let pred_ids = take(&frame.pred_ids, &pred_keep);
    let pred_confidences = take(&frame.pred_confidences, &pred_keep);
    let pred_dets = take(&frame.pred_dets, &pred_keep);
    let similarity = take_cols(&frame.similarity, &pred_keep);
    let ignore_ioa = take_rows(&frame.ignore_ioa, &pred_keep);
    let num_gt = frame.gt_ids.len();
    let num_pred = pred_ids.len();

    // Step 2: Hungarian-match predictions to all GT (distractors included); drop
    // predictions matched above threshold to a distractor-class GT. The `< thr - eps`
    // fill and the `> 0 + eps` guard copy TrackEval.
    let mut match_cols: Vec<usize> = Vec::new();
    let mut to_remove: Vec<usize> = Vec::new();
    if num_gt > 0 && num_pred > 0 {
        let threshold = protocol.distractor_iou_threshold - EPS;
        let matching: Vec<f64> = similarity
            .data()
            .iter()
            .map(|&s| {
                if s < threshold {
                    protocol.matching_fill
                } else {
                    s
                }
            })
            .collect();
        // `linear_sum_assignment(-matching_scores)`.
        let negated: Vec<f64> = matching.iter().map(|&s| -s).collect();
        let (rows, cols) = linear_sum_assignment(&negated, num_gt, num_pred, false)?;
        for (&r, &c) in rows.iter().zip(&cols) {
            if matching[r * num_pred + c] > 0.0 + EPS {
                match_cols.push(c);
                if protocol.distractor_classes.contains(&frame.gt_classes[r]) {
                    to_remove.push(c);
                }
            }
        }
    }

    // Step 3: drop unmatched predictions sitting inside an ignore region.
    if ignore_ioa.cols() > 0 {
        let ignore_threshold = protocol.ignore_iou_threshold + EPS;
        for j in (0..num_pred).filter(|j| !match_cols.contains(j)) {
            if ignore_ioa.row(j).iter().any(|&ioa| ioa > ignore_threshold) {
                to_remove.push(j);
            }
        }
    }

    // `np.delete(..., to_remove)`: a keep mask, whatever order `to_remove` is in.
    let survivors: Vec<usize> = (0..num_pred).filter(|j| !to_remove.contains(j)).collect();

    // Step 4: keep only evaluated-class GT, excluding conf-zero rows when declared.
    let gt_keep: Vec<usize> = (0..num_gt)
        .filter(|&i| {
            frame.gt_classes[i] == cls_id
                && (!protocol.drop_zero_conf_gt || frame.gt_conf[i] != 0.0)
        })
        .collect();
    Ok(PreprocessedFrame {
        gt_ids: take(&frame.gt_ids, &gt_keep),
        gt_dets: take(&frame.gt_dets, &gt_keep),
        pred_ids: take(&pred_ids, &survivors),
        pred_confidences: take(&pred_confidences, &survivors),
        pred_dets: take(&pred_dets, &survivors),
        similarity: take_rows(&take_cols(&similarity, &survivors), &gt_keep),
    })
}
