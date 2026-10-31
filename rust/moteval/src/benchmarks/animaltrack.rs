//! AnimalTrack loader, ported from `moteval/benchmarks/animaltrack.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_animaltrack` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "all";

/// `load_animaltrack(root, split)`.
pub fn load_animaltrack(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.animaltrack.load_animaltrack")
}
