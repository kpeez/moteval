//! DanceTrack loader, ported from `moteval/benchmarks/dancetrack.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_dancetrack` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "val";

/// `load_dancetrack(root, split)`.
pub fn load_dancetrack(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.dancetrack.load_dancetrack")
}
