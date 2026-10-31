//! The shared MOTChallenge-layout reader and the generic `load_motchallenge`, ported from
//! `moteval/benchmarks/motchallenge.py`.
//!
//! The formats-protocol-loaders package fills the bodies.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::model::MOTDataset;

/// The split `load_motchallenge` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "train";

/// `load_motchallenge(root, split)`: any standard MOTChallenge-layout directory, named
/// after `root`.
pub fn load_motchallenge(_root: &Path, _split: &str) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.motchallenge.load_motchallenge")
}
