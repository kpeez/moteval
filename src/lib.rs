//! moteval: multi-object tracking evaluation (HOTA, CLEAR, Identity, Count, TrackMAP,
//! J&F), ported from Python moteval with bit-identical results.
//!
//! Python moteval gives the same numbers as TrackEval commit `12c8791b`; this crate gives
//! the same bits as Python moteval. `numeric` holds every reduction order the metrics
//! need, `lsap` is scipy's assignment solver, and metrics read only `SequenceData`.

#![forbid(unsafe_code)]

pub mod benchmarks;
pub mod cli;
pub mod convert;
pub mod eval;
pub mod formats;
pub mod lsap;
pub mod metrics;
pub mod model;
pub mod numeric;
pub mod protocol;
pub mod pyfmt;
pub mod results;
pub mod rle;
pub mod similarity;

#[cfg(feature = "python")]
mod python;
