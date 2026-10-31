//! ChimpACT loader, ported from `moteval/benchmarks/chimpact.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_chimpact` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "val";

/// `load_chimpact(root, split)`.
pub fn load_chimpact(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.chimpact.load_chimpact")
}
