use std::path::{Path, PathBuf};

use crate::benchmarks::motchallenge::{MOTChallengeConfig, load_layout, max_frame_seq_length};
use crate::benchmarks::{BenchmarkError, glob_suffix};
use crate::formats::Track;
use crate::formats::pytext::{py_join, py_stem, py_str};
use crate::model::{FrameConvention, MOTDataset, Protocol};

pub const DEFAULT_SPLIT: &str = "test";

const ANIMAL_SUBSTRINGS: [&str; 4] = ["bird", "fish", "insect", "stock"];

fn gt_dir(base: &Path) -> PathBuf {
    py_join(base, "track_label")
}

fn seq_names(base: &Path, split: &str) -> Result<Vec<String>, BenchmarkError> {
    let dir = gt_dir(base);
    if !dir.is_dir() {
        return Err(BenchmarkError::SplitDirNotFound(py_str(&dir)));
    }
    let mut names: Vec<String> = glob_suffix(&dir, ".txt")?
        .iter()
        .map(|name| py_stem(name).to_owned())
        .collect();
    names.sort();
    if split == "animal" {
        names.retain(|name| ANIMAL_SUBSTRINGS.iter().any(|s| name.contains(s)));
        return Ok(names);
    }
    if split != "test" {
        return Err(BenchmarkError::UnknownSplit {
            benchmark: "gmot40",
            split: split.to_owned(),
            expected: "'test' or 'animal'",
        });
    }
    Ok(names)
}

fn gt_path(base: &Path, _split: &str, seq_name: &str) -> PathBuf {
    py_join(&gt_dir(base), format!("{seq_name}.txt"))
}

fn seq_length(
    _base: &Path,
    _split: &str,
    seq_name: &str,
    tracks: &[Track],
) -> Result<usize, BenchmarkError> {
    max_frame_seq_length(seq_name, tracks, 1)
}

pub fn load_gmot40(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let protocol = Protocol::new("gmot40", FrameConvention::new("0-indexed", 0), vec![1]);
    let config = MOTChallengeConfig {
        seq_names,
        gt_path,
        seq_length,
        ..MOTChallengeConfig::new("gmot40", protocol)
    };
    load_layout(&config, root, split)
}
