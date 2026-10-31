//! Built-in benchmark loaders, keyed by name, ported from `moteval/benchmarks/`.
//!
//! `BENCHMARKS` is the explicit index the CLI resolves `--dataset` names against;
//! `load_dataset` reads `default_data_root()/<name>` unless the caller passes a root.
//! Custom data never registers anything: it loads through `load_motchallenge` or
//! `load_mots`, or builds a `MOTDataset` directly.
//!
//! The formats-protocol-loaders package fills the bodies.

pub mod animaltrack;
pub mod bft;
pub mod chimpact;
pub mod dancetrack;
pub mod gmot40;
pub mod motchallenge;
pub mod mots20;
pub mod panaf500;
pub mod sportsmot;
pub mod uavdt;

use std::fmt;
use std::path::{Path, PathBuf};

use crate::convert::ConvertError;
use crate::formats::FormatError;
use crate::model::{MOTDataset, ModelError};

/// A loader: `(root, split) -> MOTDataset`.
pub type Loader = fn(&Path, &str) -> Result<MOTDataset, BenchmarkError>;

/// `BENCHMARKS`: name, loader, and the loader's default split, in Python's order.
pub const BENCHMARKS: &[(&str, Loader, &str)] = &[
    (
        "animaltrack",
        animaltrack::load_animaltrack,
        animaltrack::DEFAULT_SPLIT,
    ),
    ("bft", bft::load_bft, bft::DEFAULT_SPLIT),
    ("chimpact", chimpact::load_chimpact, chimpact::DEFAULT_SPLIT),
    (
        "dancetrack",
        dancetrack::load_dancetrack,
        dancetrack::DEFAULT_SPLIT,
    ),
    ("gmot40", gmot40::load_gmot40, gmot40::DEFAULT_SPLIT),
    ("mots20", mots20::load_mots20, mots20::DEFAULT_SPLIT),
    ("panaf500", panaf500::load_panaf500, panaf500::DEFAULT_SPLIT),
    (
        "sportsmot",
        sportsmot::load_sportsmot,
        sportsmot::DEFAULT_SPLIT,
    ),
    ("uavdt", uavdt::load_uavdt, uavdt::DEFAULT_SPLIT),
];

/// Errors from finding and loading a benchmark. Messages match the Python text.
#[derive(Debug)]
pub enum BenchmarkError {
    /// `MOTEVAL_DATA_ROOT must not be empty`.
    EmptyDataRoot,
    /// `unknown dataset {name!r}; available: {names}`.
    UnknownDataset(String),
    Format(FormatError),
    Model(ModelError),
    Convert(ConvertError),
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataRoot => f.write_str("MOTEVAL_DATA_ROOT must not be empty"),
            Self::UnknownDataset(name) => {
                let mut names: Vec<&str> = BENCHMARKS.iter().map(|(n, _, _)| *n).collect();
                names.sort_unstable();
                write!(
                    f,
                    "unknown dataset {}; available: {}",
                    crate::pyfmt::repr_str(name),
                    names.join(", ")
                )
            }
            Self::Format(err) => err.fmt(f),
            Self::Model(err) => err.fmt(f),
            Self::Convert(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for BenchmarkError {}

/// `default_data_root()`: `MOTEVAL_DATA_ROOT` when set (an empty or blank value is an
/// error), else `data/benchmarks` relative to the working directory.
pub fn default_data_root() -> Result<PathBuf, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.default_data_root")
}

/// `load_dataset(name, root, split)`: a built-in benchmark by name. `root` defaults to
/// `default_data_root()/<name>` and `split` to the loader's default.
pub fn load_dataset(
    _name: &str,
    _root: Option<&Path>,
    _split: Option<&str>,
) -> Result<MOTDataset, BenchmarkError> {
    todo!("formats-protocol-loaders: port moteval.benchmarks.load_dataset")
}
