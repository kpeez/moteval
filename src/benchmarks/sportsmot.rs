use std::path::Path;

use crate::benchmarks::BenchmarkError;
use crate::benchmarks::motchallenge::{MOTChallengeConfig, load_layout};
use crate::model::{FrameConvention, MOTDataset, Protocol};

pub const DEFAULT_SPLIT: &str = "val";

pub fn load_sportsmot(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let protocol = Protocol::new("sportsmot", FrameConvention::new("1-indexed", 1), vec![1]);
    load_layout(&MOTChallengeConfig::new("sportsmot", protocol), root, split)
}
