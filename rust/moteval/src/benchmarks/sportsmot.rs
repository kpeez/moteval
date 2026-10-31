//! SportsMOT loader, ported from `moteval/benchmarks/sportsmot.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_sportsmot` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "val";

/// `load_sportsmot(root, split)`.
pub fn load_sportsmot(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.sportsmot.load_sportsmot")
}
