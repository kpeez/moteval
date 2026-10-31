//! DanceTrack loader, ported from `moteval/benchmarks/dancetrack.py`: the standard
//! MOTChallenge layout, single class (pedestrian), 1-indexed.

use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::benchmarks::motchallenge::{MOTChallengeConfig, load_layout};
use crate::model::{FrameConvention, MOTDataset, Protocol};

/// The split `load_dancetrack` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "val";

/// `load_dancetrack(root, split)`.
pub fn load_dancetrack(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let protocol = Protocol::new("dancetrack", FrameConvention::new("1-indexed", 1), vec![1]);
    load_layout(
        &MOTChallengeConfig::new("dancetrack", protocol),
        root,
        split,
    )
}
