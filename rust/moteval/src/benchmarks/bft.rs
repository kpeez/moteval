//! BFT loader, ported from `moteval/benchmarks/bft.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_bft` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "val";

/// `load_bft(root, split)`.
pub fn load_bft(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.bft.load_bft")
}
