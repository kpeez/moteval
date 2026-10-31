//! PanAf500 loader, ported from `moteval/benchmarks/panaf500.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_panaf500` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "validation";

/// `load_panaf500(root, split)`.
pub fn load_panaf500(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.panaf500.load_panaf500")
}
