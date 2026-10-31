//! MOTS20 loader, ported from `moteval/benchmarks/mots20.py`.
//!
//! The formats-protocol-loaders package fills the body.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_mots20` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "train";

/// `load_mots20(root, split)`.
pub fn load_mots20(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.mots20.load_mots20")
}

/// `load_mots(root, split, class_id)`: any standard MOTS-layout directory, named after
/// `root`, evaluating `class_id` (MOTS pedestrians are class 2).
pub fn load_mots(_root: &Path, _split: &str, _class_id: i64) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.mots20.load_mots")
}
