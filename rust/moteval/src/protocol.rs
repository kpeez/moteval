//! The shared preprocessing engine, ported from `moteval/data/protocol.py`.
//!
//! `preprocess_frame` runs any `Protocol` on one frame, as TrackEval's
//! `get_preprocessed_seq_data` does (commit `12c8791b`): keep the class's predictions,
//! Hungarian-match them against all GT and drop those matched to a distractor class, drop
//! unmatched predictions inside an ignore region, then keep the class's GT (minus
//! conf-zero rows when declared). The engine reads precomputed similarity and ignore IoA,
//! so boxes and masks share it. No per-benchmark hooks.
//!
//! The formats-protocol-loaders package fills the body. The `Protocol` type itself lives
//! in `crate::model`.

use crate::lsap::LsapError;
use crate::model::{Matrix, Protocol};

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

/// Run the preprocessing engine on one frame for class `cls_id`.
pub fn preprocess_frame<D: Clone>(
    _frame: &RawFrame<D>,
    _protocol: &Protocol,
    _cls_id: i64,
) -> Result<PreprocessedFrame<D>, LsapError> {
    todo!("formats-protocol-loaders: port moteval.data.protocol.preprocess_frame")
}
