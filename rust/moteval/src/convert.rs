//! Build frame-major `SequenceData` from raw GT and prediction rows, ported from
//! `moteval/data/convert.py`.
//!
//! Raw per-frame arrays are binned under the declared `FrameConvention` (out-of-range
//! frames are errors), the preprocessing engine filters them for one class, and only the
//! survivors are densified through a map over the sorted unique raw ids (never
//! `max(ids) + 1` arrays). The mask path rejects overlapping masks and unions each
//! frame's ignore masks before the IoA test, as TrackEval's MOTS loader does.

use std::collections::BTreeMap;
use std::fmt;

use crate::formats::{MaskTrack, Track};
use crate::lsap::LsapError;
use crate::model::{
    BoxXywh, FrameConvention, Geometry, GtSequence, MaskGtSequence, Matrix, ModelError, Protocol,
    SequenceData, SequenceDataParts,
};
use crate::protocol::{PreprocessedFrame, RawFrame, preprocess_frame};
use crate::rle::{RleError, RleMask};
use crate::similarity;

/// Errors from building `SequenceData`. Messages match the Python text.
#[derive(Clone, Debug, PartialEq)]
pub enum ConvertError {
    Model(ModelError),
    Lsap(LsapError),
    Rle(RleError),
    /// `sequence {name!r}: overlapping GT masks at timestep {t}`.
    OverlappingGtMasks {
        sequence: String,
        timestep: usize,
    },
    /// `sequence {name!r}: overlapping predicted masks at timestep {t}`.
    OverlappingPredMasks {
        sequence: String,
        timestep: usize,
    },
    /// A mask row whose `img_h` or `img_w` is negative. Python hands the size to
    /// pycocotools, which fails with its own error; this message is Rust's.
    NegativeMaskSize {
        sequence: String,
        frame: i64,
        img_h: i64,
        img_w: i64,
    },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::pyfmt::repr_str;
        match self {
            Self::Model(err) => err.fmt(f),
            Self::Lsap(err) => err.fmt(f),
            Self::Rle(err) => err.fmt(f),
            Self::OverlappingGtMasks { sequence, timestep } => write!(
                f,
                "sequence {}: overlapping GT masks at timestep {timestep}",
                repr_str(sequence)
            ),
            Self::OverlappingPredMasks { sequence, timestep } => write!(
                f,
                "sequence {}: overlapping predicted masks at timestep {timestep}",
                repr_str(sequence)
            ),
            Self::NegativeMaskSize {
                sequence,
                frame,
                img_h,
                img_w,
            } => write!(
                f,
                "sequence {}: mask at frame {frame} has a negative size ({img_h}, {img_w})",
                repr_str(sequence)
            ),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<ModelError> for ConvertError {
    fn from(err: ModelError) -> Self {
        Self::Model(err)
    }
}

impl From<LsapError> for ConvertError {
    fn from(err: LsapError) -> Self {
        Self::Lsap(err)
    }
}

impl From<RleError> for ConvertError {
    fn from(err: RleError) -> Self {
        Self::Rle(err)
    }
}

/// `_bin_frames`: the rows of each timestep, in file order, under the convention.
///
/// Rows are binned in order, so the first out-of-range row raises.
fn bin_rows<'a, T>(
    rows: &'a [T],
    frame_of: impl Fn(&T) -> i64,
    num_timesteps: usize,
    convention: &FrameConvention,
) -> Result<Vec<Vec<&'a T>>, ModelError> {
    let mut bins: Vec<Vec<&T>> = vec![Vec::new(); num_timesteps];
    for row in rows {
        bins[convention.to_index(frame_of(row), num_timesteps)?].push(row);
    }
    Ok(bins)
}

/// `[t.x, t.y, t.w, t.h]` for each row.
fn boxes(rows: &[&Track]) -> Vec<BoxXywh> {
    rows.iter().map(|t| [t.x, t.y, t.w, t.h]).collect()
}

/// `build_sequence_data(gt, pred_tracks, protocol, cls_id)` for boxes.
///
/// GT rows are binned first, then predictions, then ignore regions; the first
/// out-of-range frame in that order raises.
pub fn build_sequence_data(
    gt: &GtSequence,
    pred_tracks: &[Track],
    protocol: &Protocol,
    cls_id: i64,
) -> Result<SequenceData, ConvertError> {
    let convention = &protocol.frame_convention;
    let n = gt.num_timesteps;
    let gt_bins = bin_rows(&gt.tracks, |t| t.frame, n, convention)?;
    let pred_bins = bin_rows(pred_tracks, |t| t.frame, n, convention)?;
    let ignore_bins = bin_rows(&gt.ignore_regions, |t| t.frame, n, convention)?;

    let mut frames = Vec::with_capacity(n);
    for t in 0..n {
        let (gt_rows, pred_rows) = (&gt_bins[t], &pred_bins[t]);
        let gt_boxes = boxes(gt_rows);
        let pred_boxes = boxes(pred_rows);
        let ignore_boxes = boxes(&ignore_bins[t]);
        let raw = RawFrame {
            gt_ids: gt_rows.iter().map(|r| r.track_id).collect(),
            gt_classes: gt_rows.iter().map(|r| r.class_id).collect(),
            gt_conf: gt_rows.iter().map(|r| r.conf).collect(),
            similarity: similarity::box_iou(&gt_boxes, &pred_boxes),
            ignore_ioa: similarity::box_ioa(&pred_boxes, &ignore_boxes),
            gt_dets: gt_boxes,
            pred_ids: pred_rows.iter().map(|r| r.track_id).collect(),
            pred_classes: pred_rows.iter().map(|r| r.class_id).collect(),
            pred_confidences: pred_rows.iter().map(|r| r.conf).collect(),
            pred_dets: pred_boxes,
        };
        frames.push(preprocess_frame(&raw, protocol, cls_id)?);
    }
    let (gt_dets, pred_dets) = split_dets(&mut frames);
    freeze(
        &gt.name,
        n,
        frames,
        Geometry::Boxes {
            gt: gt_dets,
            pred: pred_dets,
        },
    )
}

/// The mask geometry `build_mask_sequence_data` reads, as `moteval/data/similarity.py`
/// computes it.
///
/// `build_mask_sequence_data` uses the pycocotools port in `crate::similarity`. The trait
/// exists so the conversion can be tested apart from the RLE code.
#[doc(hidden)]
pub trait MaskGeometry {
    /// `masks_overlap(rles)`.
    fn masks_overlap(&self, masks: &[RleMask]) -> Result<bool, RleError>;
    /// `mask_ioa(preds, [merge_masks(ignore)])`: an `(n, 1)` matrix; `ignore` is not empty.
    fn ignore_ioa(&self, preds: &[RleMask], ignore: &[RleMask]) -> Result<Matrix, RleError>;
    /// `mask_iou(gt, preds)`.
    fn mask_iou(&self, gt: &[RleMask], preds: &[RleMask]) -> Result<Matrix, RleError>;
}

/// The pycocotools port: what Python moteval calls.
struct Pycocotools;

impl MaskGeometry for Pycocotools {
    fn masks_overlap(&self, masks: &[RleMask]) -> Result<bool, RleError> {
        similarity::masks_overlap(masks)
    }

    fn ignore_ioa(&self, preds: &[RleMask], ignore: &[RleMask]) -> Result<Matrix, RleError> {
        let merged = similarity::merge_masks(ignore)?;
        similarity::mask_ioa(preds, &[merged])
    }

    fn mask_iou(&self, gt: &[RleMask], preds: &[RleMask]) -> Result<Matrix, RleError> {
        similarity::mask_iou(gt, preds)
    }
}

/// `_rle(track)`: `{"size": [img_h, img_w], "counts": rle.encode()}`.
fn rle_of(sequence: &str, row: &MaskTrack) -> Result<RleMask, ConvertError> {
    let negative = || ConvertError::NegativeMaskSize {
        sequence: sequence.to_owned(),
        frame: row.frame,
        img_h: row.img_h,
        img_w: row.img_w,
    };
    let height = usize::try_from(row.img_h).map_err(|_| negative())?;
    let width = usize::try_from(row.img_w).map_err(|_| negative())?;
    Ok(RleMask::from_compressed(height, width, row.rle.as_bytes())?)
}

/// `build_mask_sequence_data(gt, pred_tracks, protocol, cls_id)` for masks.
pub fn build_mask_sequence_data(
    gt: &MaskGtSequence,
    pred_tracks: &[MaskTrack],
    protocol: &Protocol,
    cls_id: i64,
) -> Result<SequenceData, ConvertError> {
    build_mask_sequence_data_with(&Pycocotools, gt, pred_tracks, protocol, cls_id)
}

/// `build_mask_sequence_data` with the mask geometry supplied by the caller.
#[doc(hidden)]
pub fn build_mask_sequence_data_with(
    geometry: &dyn MaskGeometry,
    gt: &MaskGtSequence,
    pred_tracks: &[MaskTrack],
    protocol: &Protocol,
    cls_id: i64,
) -> Result<SequenceData, ConvertError> {
    let convention = &protocol.frame_convention;
    let n = gt.num_timesteps;
    let name = gt.name.as_str();
    let gt_bins = bin_rows(&gt.tracks, |t| t.frame, n, convention)?;
    let pred_bins = bin_rows(pred_tracks, |t| t.frame, n, convention)?;
    let ignore_bins = bin_rows(&gt.ignore_regions, |t| t.frame, n, convention)?;

    let mut frames = Vec::with_capacity(n);
    for t in 0..n {
        let to_rles = |rows: &[&MaskTrack]| -> Result<Vec<RleMask>, ConvertError> {
            rows.iter().map(|row| rle_of(name, row)).collect()
        };
        // Python builds every RLE dict while binning and pycocotools first reads them in
        // the overlap checks below, GT and ignore masks before predictions.
        let gt_rles = to_rles(&gt_bins[t])?;
        let ignore_rles = to_rles(&ignore_bins[t])?;
        // TrackEval rejects overlapping masks within a frame: GT dets share the frame
        // with the ignore region; predictions are checked on their own.
        let gt_and_ignore: Vec<RleMask> = gt_rles.iter().chain(&ignore_rles).cloned().collect();
        if geometry.masks_overlap(&gt_and_ignore)? {
            return Err(ConvertError::OverlappingGtMasks {
                sequence: name.to_owned(),
                timestep: t,
            });
        }
        let pred_rles = to_rles(&pred_bins[t])?;
        if geometry.masks_overlap(&pred_rles)? {
            return Err(ConvertError::OverlappingPredMasks {
                sequence: name.to_owned(),
                timestep: t,
            });
        }
        // Ignore masks merge into one union RLE, exactly as upstream MOTS does; with no
        // ignore rows the IoA matrix is (n, 0) and the engine skips it.
        let ignore_ioa = if ignore_rles.is_empty() {
            Matrix::zeros(pred_rles.len(), 0)
        } else {
            geometry.ignore_ioa(&pred_rles, &ignore_rles)?
        };
        let (gt_rows, pred_rows) = (&gt_bins[t], &pred_bins[t]);
        let raw = RawFrame {
            gt_ids: gt_rows.iter().map(|r| r.track_id).collect(),
            gt_classes: gt_rows.iter().map(|r| r.class_id).collect(),
            gt_conf: vec![1.0; gt_rows.len()],
            similarity: geometry.mask_iou(&gt_rles, &pred_rles)?,
            ignore_ioa,
            gt_dets: gt_rles,
            pred_ids: pred_rows.iter().map(|r| r.track_id).collect(),
            pred_classes: pred_rows.iter().map(|r| r.class_id).collect(),
            pred_confidences: vec![1.0; pred_rows.len()],
            pred_dets: pred_rles,
        };
        frames.push(preprocess_frame(&raw, protocol, cls_id)?);
    }
    let (gt_dets, pred_dets) = split_dets(&mut frames);
    freeze(
        name,
        n,
        frames,
        Geometry::Masks {
            gt: gt_dets,
            pred: pred_dets,
        },
    )
}

/// Move each frame's surviving detections out, for the geometry.
fn split_dets<D>(frames: &mut [PreprocessedFrame<D>]) -> (Vec<Vec<D>>, Vec<Vec<D>>) {
    frames
        .iter_mut()
        .map(|f| {
            (
                std::mem::take(&mut f.gt_dets),
                std::mem::take(&mut f.pred_dets),
            )
        })
        .unzip()
}

/// `_densify`: raw id to its index among the sorted unique raw ids.
fn densify<'a>(ids: impl Iterator<Item = &'a i64>) -> BTreeMap<i64, usize> {
    let mut map: BTreeMap<i64, usize> = ids.map(|&id| (id, 0)).collect();
    for (dense, slot) in map.values_mut().enumerate() {
        *slot = dense;
    }
    map
}

/// `_freeze`: densify the surviving ids and assemble the `SequenceData`.
fn freeze<D>(
    name: &str,
    num_timesteps: usize,
    frames: Vec<PreprocessedFrame<D>>,
    geometry: Geometry,
) -> Result<SequenceData, ConvertError> {
    let gt_map = densify(frames.iter().flat_map(|f| &f.gt_ids));
    let pred_map = densify(frames.iter().flat_map(|f| &f.pred_ids));
    let parts = SequenceDataParts {
        name: name.to_owned(),
        num_timesteps,
        num_gt_ids: gt_map.len(),
        num_pred_ids: pred_map.len(),
        num_gt_dets: frames.iter().map(|f| f.gt_ids.len()).sum(),
        num_pred_dets: frames.iter().map(|f| f.pred_ids.len()).sum(),
        gt_ids: frames
            .iter()
            .map(|f| f.gt_ids.iter().map(|id| gt_map[id]).collect())
            .collect(),
        pred_ids: frames
            .iter()
            .map(|f| f.pred_ids.iter().map(|id| pred_map[id]).collect())
            .collect(),
        pred_confidences: frames.iter().map(|f| f.pred_confidences.clone()).collect(),
        geometry,
        similarity: frames.into_iter().map(|f| f.similarity).collect(),
    };
    Ok(SequenceData::new(parts)?)
}
