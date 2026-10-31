//! GMOT-40 loader, ported from `moteval/benchmarks/gmot40.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_gmot40` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "test";

/// `load_gmot40(root, split)`.
pub fn load_gmot40(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.gmot40.load_gmot40")
}
