//! AnimalTrack loader, ported from `moteval/benchmarks/animaltrack.py`.
//!
//! MOTChallenge-style GT rows in `<root>/gt_all/<seq>_gt.txt`, no `seqinfo.ini` (the
//! sequence length is the last annotated frame), 1-indexed. "train" and "test" list
//! their sequences in `<root>/train_test_splits/videos_<split>.txt` (one `<seq>.mp4` per
//! line); "all" is every `gt_all/*_gt.txt`.

use std::path::{Path, PathBuf};

use crate::benchmarks::motchallenge::{MOTChallengeConfig, load_layout, max_frame_seq_length};
use crate::benchmarks::{BenchmarkError, glob_suffix, read_text};
use crate::formats::Track;
use crate::formats::pytext::{py_join, py_str, splitlines, strip};
use crate::model::{FrameConvention, MOTDataset, Protocol};

/// The split `load_animaltrack` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "all";

fn seq_names(base: &Path, split: &str) -> Result<Vec<String>, BenchmarkError> {
    if split == "all" {
        let gt_dir = py_join(base, "gt_all");
        if !gt_dir.is_dir() {
            return Err(BenchmarkError::SplitDirNotFound(py_str(&gt_dir)));
        }
        let mut names: Vec<String> = glob_suffix(&gt_dir, "_gt.txt")?
            .iter()
            .map(|name| name.strip_suffix("_gt.txt").unwrap_or(name).to_owned())
            .collect();
        names.sort();
        return Ok(names);
    }
    if split != "train" && split != "test" {
        return Err(BenchmarkError::UnknownSplit {
            benchmark: "animaltrack",
            split: split.to_owned(),
            expected: "'all', 'train', or 'test'",
        });
    }
    let split_file = py_join(base, format!("train_test_splits/videos_{split}.txt"));
    if !split_file.is_file() {
        return Err(BenchmarkError::SplitFileNotFound {
            split: split.to_owned(),
            path: py_str(&split_file),
        });
    }
    Ok(splitlines(&read_text(&split_file)?)
        .into_iter()
        .map(strip)
        .filter(|line| !line.is_empty())
        .map(|line| line.strip_suffix(".mp4").unwrap_or(line).to_owned())
        .collect())
}

fn gt_path(base: &Path, _split: &str, seq_name: &str) -> PathBuf {
    py_join(&py_join(base, "gt_all"), format!("{seq_name}_gt.txt"))
}

fn seq_length(
    _base: &Path,
    _split: &str,
    seq_name: &str,
    tracks: &[Track],
) -> Result<usize, BenchmarkError> {
    max_frame_seq_length(seq_name, tracks, 0)
}

/// `load_animaltrack(root, split)`.
pub fn load_animaltrack(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let protocol = Protocol::new("animaltrack", FrameConvention::new("1-indexed", 1), vec![1]);
    let config = MOTChallengeConfig {
        seq_names,
        gt_path,
        seq_length,
        ..MOTChallengeConfig::new("animaltrack", protocol)
    };
    load_layout(&config, root, split)
}
