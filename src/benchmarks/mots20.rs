use std::path::Path;

use crate::benchmarks::motchallenge::read_seq_length;
use crate::benchmarks::{BenchmarkError, sorted_subdirs};
use crate::formats::pytext::{py_join, py_name, py_path, py_str};
use crate::formats::read_mots;
use crate::model::{FrameConvention, MOTDataset, MaskGtSequence, Protocol, Sequence};

pub const DEFAULT_SPLIT: &str = "train";

pub const MOTS20_IGNORE_CLASS: i64 = 10;

// TrackEval MOTS: below-threshold matches fill with -10000, and GT is never conf-filtered.
pub fn mots20_protocol() -> Protocol {
    Protocol {
        drop_zero_conf_gt: false,
        matching_fill: -10000.0,
        ..Protocol::new("mots20", FrameConvention::new("1-indexed", 1), vec![2])
    }
}

fn load_sequence(
    base: &Path,
    split: &str,
    seq_name: &str,
) -> Result<MaskGtSequence, BenchmarkError> {
    let seq_dir = py_join(&py_join(base, split), seq_name);
    let gt_path = py_join(&seq_dir, "gt/gt.txt");
    if !gt_path.is_file() {
        return Err(BenchmarkError::MissingFile {
            file: "gt.txt",
            sequence: seq_name.to_owned(),
            path: py_str(&gt_path),
        });
    }
    let (ignore_regions, tracks) = read_mots(&gt_path)?
        .into_iter()
        .partition(|t| t.class_id == MOTS20_IGNORE_CLASS);
    let num_timesteps = read_seq_length(&seq_dir)?;
    Ok(MaskGtSequence {
        name: seq_name.to_owned(),
        num_timesteps,
        tracks,
        ignore_regions,
    })
}

fn load_split(
    base: &Path,
    split: &str,
    name: String,
    protocol: Protocol,
) -> Result<MOTDataset, BenchmarkError> {
    let split_dir = py_join(base, split);
    if !split_dir.is_dir() {
        return Err(BenchmarkError::SplitDirNotFound(py_str(&split_dir)));
    }
    let sequences = sorted_subdirs(&split_dir)?
        .iter()
        .map(|seq_name| load_sequence(base, split, seq_name).map(Sequence::Masks))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MOTDataset {
        name,
        split: split.to_owned(),
        sequences,
        protocol,
    })
}

pub fn load_mots20(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    load_split(
        &py_path(root),
        split,
        "mots20".to_owned(),
        mots20_protocol(),
    )
}

pub fn load_mots(root: &Path, split: &str, class_id: i64) -> Result<MOTDataset, BenchmarkError> {
    let protocol = Protocol {
        name: "mots".to_owned(),
        eval_classes: vec![class_id],
        ..mots20_protocol()
    };
    load_split(&py_path(root), split, py_name(root), protocol)
}
