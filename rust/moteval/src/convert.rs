//! Build frame-major `SequenceData` from raw GT and prediction rows, ported from
//! `moteval/data/convert.py`.
//!
//! Raw per-frame arrays are binned under the declared `FrameConvention` (out-of-range
//! frames are errors), the preprocessing engine filters them for one class, and only the
//! survivors are densified through a map over the sorted unique raw ids (never
//! `max(ids) + 1` arrays). The mask path rejects overlapping masks and unions each
//! frame's ignore masks before the IoA test, as TrackEval's MOTS loader does.
//!
//! The formats-protocol-loaders package fills these bodies.

use std::fmt;

use crate::formats::{MaskTrack, Track};
use crate::lsap::LsapError;
use crate::model::{GtSequence, MaskGtSequence, ModelError, Protocol, SequenceData};
use crate::rle::RleError;

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
        }
    }
}

impl std::error::Error for ConvertError {}

/// `build_sequence_data(gt, pred_tracks, protocol, cls_id)` for boxes.
pub fn build_sequence_data(
    _gt: &GtSequence,
    _pred_tracks: &[Track],
    _protocol: &Protocol,
    _cls_id: i64,
) -> Result<SequenceData, ConvertError> {
    todo!("formats-protocol-loaders: port moteval.data.convert.build_sequence_data")
}

/// `build_mask_sequence_data(gt, pred_tracks, protocol, cls_id)` for masks.
pub fn build_mask_sequence_data(
    _gt: &MaskGtSequence,
    _pred_tracks: &[MaskTrack],
    _protocol: &Protocol,
    _cls_id: i64,
) -> Result<SequenceData, ConvertError> {
    todo!("formats-protocol-loaders: port moteval.data.convert.build_mask_sequence_data")
}
