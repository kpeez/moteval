//! The shared MOTChallenge-layout reader and the generic `load_motchallenge`, ported from
//! `moteval/benchmarks/motchallenge.py`.
//!
//! Default layout: `<root>/<split>/<seq>/gt/gt.txt` plus `<root>/<split>/<seq>/seqinfo.ini`
//! (`[Sequence]` section, `seqLength=N`). Sequences are the split directory's
//! subdirectories, sorted. GT rows are read without the class column and stamped with
//! `MOTChallengeConfig::class_id`. Benchmarks whose layout deviates (BFT, AnimalTrack,
//! GMOT-40, UAVDT) override the four hooks of `MOTChallengeConfig`.

use std::fs;
use std::path::{Path, PathBuf};

use super::ini::{ConfigParser, decode_chunked, posix_path_repr};
use super::{BenchmarkError, PyException, sorted_subdirs};
use crate::formats::pytext::{PyInt, py_int, py_join, py_name, py_path, py_str};
use crate::formats::{Track, read_mot};
use crate::model::{FrameConvention, GtSequence, MOTDataset, Protocol, Sequence};

/// The split `load_motchallenge` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "train";

/// `(base, split) -> sequence names`.
pub type SeqNamesFn = fn(&Path, &str) -> Result<Vec<String>, BenchmarkError>;
/// `(base, split, sequence) -> GT file`.
pub type GtPathFn = fn(&Path, &str, &str) -> PathBuf;
/// `(base, split, sequence, GT tracks) -> num_timesteps`.
pub type SeqLengthFn = fn(&Path, &str, &str, &[Track]) -> Result<usize, BenchmarkError>;
/// `(base, split, sequence) -> ignore-region file, if the benchmark has one`.
pub type IgnorePathFn = fn(&Path, &str, &str) -> Option<PathBuf>;

/// Per-benchmark configuration for the generic MOTChallenge adapter (`MOTChallengeConfig`).
#[derive(Clone, Debug)]
pub struct MOTChallengeConfig {
    pub name: String,
    pub protocol: Protocol,
    /// The class every GT `Track` is tagged with (the adapter reads no class column).
    pub class_id: i64,
    pub seq_names: SeqNamesFn,
    pub gt_path: GtPathFn,
    pub seq_length: SeqLengthFn,
    pub ignore_path: IgnorePathFn,
}

impl MOTChallengeConfig {
    /// The standard layout with class 1, as `MOTChallengeConfig(name=..., protocol=...)`.
    pub fn new(name: impl Into<String>, protocol: Protocol) -> Self {
        Self {
            name: name.into(),
            protocol,
            class_id: 1,
            seq_names: default_seq_names,
            gt_path: default_gt_path,
            seq_length: seqinfo_seq_length,
            ignore_path: default_ignore_path,
        }
    }
}

fn default_seq_names(base: &Path, split: &str) -> Result<Vec<String>, BenchmarkError> {
    let split_dir = py_join(base, split);
    if !split_dir.is_dir() {
        return Err(BenchmarkError::SplitDirNotFound(py_str(&split_dir)));
    }
    sorted_subdirs(&split_dir)
}

fn default_gt_path(base: &Path, split: &str, seq_name: &str) -> PathBuf {
    py_join(&py_join(&py_join(base, split), seq_name), "gt/gt.txt")
}

fn seqinfo_seq_length(
    base: &Path,
    split: &str,
    seq_name: &str,
    _tracks: &[Track],
) -> Result<usize, BenchmarkError> {
    read_seq_length(&py_join(&py_join(base, split), seq_name))
}

fn default_ignore_path(_base: &Path, _split: &str, _seq_name: &str) -> Option<PathBuf> {
    None
}

/// A sequence length Python computed, as `usize`.
fn to_num_timesteps(value: i128, sequence: &str) -> Result<usize, BenchmarkError> {
    usize::try_from(value).map_err(|_| {
        BenchmarkError::Unsupported(format!(
            "sequence {}: sequence length {value} is negative or too large",
            crate::pyfmt::repr_str(sequence)
        ))
    })
}

/// `max_frame_seq_length(seq_name, tracks, offset)`: the last annotated frame plus
/// `offset`, for benchmarks without a `seqinfo.ini`. Empty GT is an error.
pub fn max_frame_seq_length(
    seq_name: &str,
    tracks: &[Track],
    offset: i64,
) -> Result<usize, BenchmarkError> {
    let Some(last) = tracks.iter().map(|t| t.frame).max() else {
        return Err(BenchmarkError::EmptyGt(seq_name.to_owned()));
    };
    to_num_timesteps(i128::from(last) + i128::from(offset), seq_name)
}

/// `read_seq_length(seq_dir)`: `[Sequence] seqLength` from `<seq_dir>/seqinfo.ini`, read
/// with `configparser.ConfigParser`.
pub fn read_seq_length(seq_dir: &Path) -> Result<usize, BenchmarkError> {
    let seq_dir = py_path(seq_dir);
    let sequence = py_name(&seq_dir);
    let seqinfo_path = py_join(&seq_dir, "seqinfo.ini");
    let path = py_str(&seqinfo_path);
    if !seqinfo_path.is_file() {
        return Err(BenchmarkError::MissingFile {
            file: "seqinfo.ini",
            sequence,
            path,
        });
    }
    let malformed = |detail: String| BenchmarkError::MalformedSeqinfo {
        sequence: sequence.clone(),
        path: path.clone(),
        detail,
    };
    // `parser.read` skips a file it cannot open.
    let parser = match fs::read(&seqinfo_path) {
        Err(_) => ConfigParser::default(),
        Ok(bytes) => {
            let text = decode_chunked(&bytes).map_err(|message| {
                BenchmarkError::python(PyException::UnicodeDecodeError, message)
            })?;
            ConfigParser::read(&text, &posix_path_repr(&path)).map_err(malformed)?
        }
    };
    if !parser.has_option("Sequence", "seqLength") {
        return Err(malformed("missing [Sequence] seqLength".to_owned()));
    }
    let value = parser
        .get("Sequence", "seqLength")
        .map_err(|message| BenchmarkError::python(PyException::InterpolationError, message))?;
    match py_int(&value) {
        Ok(PyInt::Value(v)) => to_num_timesteps(i128::from(v), &sequence),
        Ok(PyInt::OutOfRange { .. }) => Err(BenchmarkError::Unsupported(format!(
            "seqinfo.ini for sequence {} at {path}: seqLength {} is outside the 64-bit range",
            crate::pyfmt::repr_str(&sequence),
            crate::pyfmt::repr_str(&value)
        ))),
        Err(_) => Err(malformed("seqLength is not an integer".to_owned())),
    }
}

/// `_load_sequence(base, split, seq_name, config)`.
fn load_sequence(
    base: &Path,
    split: &str,
    seq_name: &str,
    config: &MOTChallengeConfig,
) -> Result<GtSequence, BenchmarkError> {
    let gt_path = py_path(&(config.gt_path)(base, split, seq_name));
    if !gt_path.is_file() {
        return Err(BenchmarkError::MissingFile {
            file: "gt.txt",
            sequence: seq_name.to_owned(),
            path: py_str(&gt_path),
        });
    }
    let mut tracks = read_mot(&gt_path, false)?;
    for track in &mut tracks {
        track.class_id = config.class_id;
    }
    let num_timesteps = (config.seq_length)(base, split, seq_name, &tracks)?;
    let ignore_regions = match (config.ignore_path)(base, split, seq_name) {
        Some(path) if py_path(&path).is_file() => read_mot(&path, false)?,
        _ => Vec::new(),
    };
    Ok(GtSequence {
        name: seq_name.to_owned(),
        num_timesteps,
        tracks,
        ignore_regions,
    })
}

/// `load_layout(config, root, split)`: a MOTChallenge-layout split as a `MOTDataset`.
pub fn load_layout(
    config: &MOTChallengeConfig,
    root: &Path,
    split: &str,
) -> Result<MOTDataset, BenchmarkError> {
    let base = py_path(root);
    let seq_names = (config.seq_names)(&base, split)?;
    let sequences = seq_names
        .iter()
        .map(|name| load_sequence(&base, split, name, config).map(Sequence::Boxes))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MOTDataset {
        name: config.name.clone(),
        split: split.to_owned(),
        sequences,
        protocol: config.protocol.clone(),
    })
}

/// `MOTCHALLENGE_PROTOCOL`: 1-indexed frames, class 1, Python's defaults otherwise.
pub fn motchallenge_protocol() -> Protocol {
    Protocol::new(
        "motchallenge",
        FrameConvention::new("1-indexed", 1),
        vec![1],
    )
}

/// `load_motchallenge(root, split)`: any standard MOTChallenge-layout directory, named
/// after `root`.
pub fn load_motchallenge(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let config = MOTChallengeConfig::new(py_name(root), motchallenge_protocol());
    load_layout(&config, root, split)
}
