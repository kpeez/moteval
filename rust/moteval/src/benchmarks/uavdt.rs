//! UAVDT loader, ported from `moteval/benchmarks/uavdt.py`.
//!
//! MOTChallenge-style GT rows in `<root>/UAV-benchmark-MOTD_v1.0/GT/<seq>_gt.txt`, with a
//! sibling `<seq>_gt_ignore.txt` of crowd-ignore regions (absent means none). No
//! `seqinfo.ini`: the sequence length is the last annotated frame. The only split is
//! "all"; `*_gt_ignore.txt` and `*_gt_whole.txt` do not end in exactly `_gt.txt`.

use std::path::{Path, PathBuf};

use crate::benchmarks::motchallenge::{MOTChallengeConfig, load_layout, max_frame_seq_length};
use crate::benchmarks::{BenchmarkError, glob_suffix};
use crate::formats::Track;
use crate::formats::pytext::{py_join, py_str};
use crate::model::{FrameConvention, MOTDataset, Protocol};

/// The split `load_uavdt` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "all";

fn gt_dir(base: &Path) -> PathBuf {
    py_join(base, "UAV-benchmark-MOTD_v1.0/GT")
}

fn seq_names(base: &Path, split: &str) -> Result<Vec<String>, BenchmarkError> {
    if split != "all" {
        return Err(BenchmarkError::UnknownSplit {
            benchmark: "uavdt",
            split: split.to_owned(),
            expected: "'all'",
        });
    }
    let dir = gt_dir(base);
    if !dir.is_dir() {
        return Err(BenchmarkError::SplitDirNotFound(py_str(&dir)));
    }
    let mut names: Vec<String> = glob_suffix(&dir, "_gt.txt")?
        .iter()
        .map(|name| name.strip_suffix("_gt.txt").unwrap_or(name).to_owned())
        .collect();
    names.sort();
    Ok(names)
}

fn gt_path(base: &Path, _split: &str, seq_name: &str) -> PathBuf {
    py_join(&gt_dir(base), format!("{seq_name}_gt.txt"))
}

fn ignore_path(base: &Path, _split: &str, seq_name: &str) -> Option<PathBuf> {
    Some(py_join(&gt_dir(base), format!("{seq_name}_gt_ignore.txt")))
}

fn seq_length(
    _base: &Path,
    _split: &str,
    seq_name: &str,
    tracks: &[Track],
) -> Result<usize, BenchmarkError> {
    max_frame_seq_length(seq_name, tracks, 0)
}

/// `load_uavdt(root, split)`.
pub fn load_uavdt(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let protocol = Protocol::new("uavdt", FrameConvention::new("1-indexed", 1), vec![1]);
    let config = MOTChallengeConfig {
        seq_names,
        gt_path,
        seq_length,
        ignore_path,
        ..MOTChallengeConfig::new("uavdt", protocol)
    };
    load_layout(&config, root, split)
}
