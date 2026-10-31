//! UAVDT loader, ported from `moteval/benchmarks/uavdt.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_uavdt` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "all";

/// `load_uavdt(root, split)`.
pub fn load_uavdt(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.uavdt.load_uavdt")
}
