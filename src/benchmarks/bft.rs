use std::path::{Path, PathBuf};

use crate::benchmarks::motchallenge::{MOTChallengeConfig, load_layout, max_frame_seq_length};
use crate::benchmarks::{BenchmarkError, glob_suffix};
use crate::formats::Track;
use crate::formats::pytext::{py_join, py_stem, py_str};
use crate::model::{FrameConvention, MOTDataset, Protocol};

pub const DEFAULT_SPLIT: &str = "val";

fn ann_dir(base: &Path, split: &str) -> PathBuf {
    py_join(&py_join(base, "annotations_mot"), split)
}

fn seq_names(base: &Path, split: &str) -> Result<Vec<String>, BenchmarkError> {
    let dir = ann_dir(base, split);
    if !dir.is_dir() {
        return Err(BenchmarkError::SplitDirNotFound(py_str(&dir)));
    }
    let mut names: Vec<String> = glob_suffix(&dir, ".txt")?
        .iter()
        .map(|name| py_stem(name).to_owned())
        .collect();
    names.sort();
    Ok(names)
}

fn gt_path(base: &Path, split: &str, seq_name: &str) -> PathBuf {
    py_join(&ann_dir(base, split), format!("{seq_name}.txt"))
}

fn seq_length(
    _base: &Path,
    _split: &str,
    seq_name: &str,
    tracks: &[Track],
) -> Result<usize, BenchmarkError> {
    max_frame_seq_length(seq_name, tracks, 0)
}

pub fn load_bft(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let protocol = Protocol::new("bft", FrameConvention::new("1-indexed", 1), vec![1]);
    let config = MOTChallengeConfig {
        seq_names,
        gt_path,
        seq_length,
        ..MOTChallengeConfig::new("bft", protocol)
    };
    load_layout(&config, root, split)
}
