//! Built-in benchmark loaders, keyed by name, ported from `moteval/benchmarks/`.
//!
//! `BENCHMARKS` is the explicit index the CLI resolves `--dataset` names against;
//! `load_dataset` reads `default_data_root()/<name>` unless the caller passes a root.
//! Custom data never registers anything: it loads through `load_motchallenge` or
//! `load_mots`, or builds a `MOTDataset` directly.
//!
//! Loaders find files as pathlib does (`py_path`, dot files match `*`), read text as
//! Python does, and fail with Python's message for every input Python rejects. Where
//! Python accepts a value Rust cannot hold (an integer beyond `i64`, a negative sequence
//! length, a non-numeric JSON coordinate) the loader fails with `Unsupported`.

pub mod animaltrack;
pub mod bft;
pub mod chimpact;
pub mod dancetrack;
pub mod gmot40;
pub(crate) mod ini;
pub mod motchallenge;
pub mod mots20;
pub mod panaf500;
pub(crate) mod pyjson;
pub mod sportsmot;
pub mod uavdt;

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::convert::ConvertError;
use crate::formats::FormatError;
use crate::formats::pytext::{self, py_join, py_path, py_str};
use crate::model::{MOTDataset, ModelError};
use crate::pyfmt::repr_str;

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

/// The Python exception a `BenchmarkError::Python` stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyException {
    /// `OSError` and its subclasses (`FileNotFoundError`, `IsADirectoryError`, ...).
    OsError,
    /// `UnicodeDecodeError` (a `ValueError`).
    UnicodeDecodeError,
    /// `json.JSONDecodeError` (a `ValueError`).
    JsonDecodeError,
    /// Any other `ValueError` (unpacking, `zip(strict=True)`, `int()`).
    ValueError,
    KeyError,
    TypeError,
    /// `configparser.InterpolationError` (not a `ValueError`).
    InterpolationError,
}

/// Errors from finding and loading a benchmark. Messages match the Python text.
#[derive(Clone, Debug, PartialEq)]
pub enum BenchmarkError {
    /// `MOTEVAL_DATA_ROOT must not be empty`.
    EmptyDataRoot,
    /// `unknown dataset {name!r}; available: {names}`.
    UnknownDataset(String),
    /// `split directory not found: {path}`.
    SplitDirNotFound(String),
    /// `labels directory not found: {path}` (ChimpACT).
    LabelsDirNotFound(String),
    /// `unknown {benchmark} split {split!r}; expected {expected}`.
    UnknownSplit {
        benchmark: &'static str,
        split: String,
        expected: &'static str,
    },
    /// `animaltrack split {split!r} not found at {path}`.
    SplitFileNotFound {
        split: String,
        path: String,
    },
    /// `missing {file} for sequence {sequence!r} at {path}` (`gt.txt`, `seqinfo.ini`).
    MissingFile {
        file: &'static str,
        sequence: String,
        path: String,
    },
    /// `malformed seqinfo.ini for sequence {sequence!r} at {path}: {detail}`.
    MalformedSeqinfo {
        sequence: String,
        path: String,
        detail: String,
    },
    /// `cannot derive sequence length for empty gt: {sequence!r}`.
    EmptyGt(String),
    /// An exception Python raises from a library call, with its text.
    Python {
        exception: PyException,
        message: String,
    },
    /// A value Python accepts that Rust cannot hold. The message is Rust's own.
    Unsupported(String),
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
                    repr_str(name),
                    names.join(", ")
                )
            }
            Self::SplitDirNotFound(path) => write!(f, "split directory not found: {path}"),
            Self::LabelsDirNotFound(path) => write!(f, "labels directory not found: {path}"),
            Self::UnknownSplit {
                benchmark,
                split,
                expected,
            } => write!(
                f,
                "unknown {benchmark} split {}; expected {expected}",
                repr_str(split)
            ),
            Self::SplitFileNotFound { split, path } => write!(
                f,
                "animaltrack split {} not found at {path}",
                repr_str(split)
            ),
            Self::MissingFile {
                file,
                sequence,
                path,
            } => write!(
                f,
                "missing {file} for sequence {} at {path}",
                repr_str(sequence)
            ),
            Self::MalformedSeqinfo {
                sequence,
                path,
                detail,
            } => write!(
                f,
                "malformed seqinfo.ini for sequence {} at {path}: {detail}",
                repr_str(sequence)
            ),
            Self::EmptyGt(sequence) => write!(
                f,
                "cannot derive sequence length for empty gt: {}",
                repr_str(sequence)
            ),
            Self::Python { message, .. } | Self::Unsupported(message) => f.write_str(message),
            Self::Format(err) => err.fmt(f),
            Self::Model(err) => err.fmt(f),
            Self::Convert(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for BenchmarkError {}

impl From<FormatError> for BenchmarkError {
    fn from(err: FormatError) -> Self {
        Self::Format(err)
    }
}

impl BenchmarkError {
    fn python(exception: PyException, message: impl Into<String>) -> Self {
        Self::Python {
            exception,
            message: message.into(),
        }
    }

    fn os(err: &std::io::Error, path: &Path) -> Self {
        Self::python(PyException::OsError, pytext::os_error_message(err, path))
    }
}

/// `Path.read_text()`, failing as Python's `OSError` or `UnicodeDecodeError`.
fn read_text(path: &Path) -> Result<String, BenchmarkError> {
    crate::formats::read_text(path).map_err(|err| match err {
        FormatError::Decode(message) => {
            BenchmarkError::python(PyException::UnicodeDecodeError, message)
        }
        other => BenchmarkError::python(PyException::OsError, other.to_string()),
    })
}

/// `json.loads(path.read_text())`.
fn read_json(path: &Path) -> Result<pyjson::Json, BenchmarkError> {
    let text = read_text(path)?;
    pyjson::loads(&text).map_err(|err| match err {
        pyjson::JsonError::Decode(message) => {
            BenchmarkError::python(PyException::JsonDecodeError, message)
        }
        pyjson::JsonError::TooManyDigits(message) => {
            BenchmarkError::python(PyException::ValueError, message)
        }
        pyjson::JsonError::Unsupported(message) => BenchmarkError::Unsupported(message),
    })
}

/// A JSON value a loader stores in an integer `Track` field. Python stores any value and
/// fails (or not) later; Rust reads an `int` (or `bool`, an `int` in Python) that fits.
fn json_i64(value: &pyjson::Json, field: &str, source: &Path) -> Result<i64, BenchmarkError> {
    value
        .number()
        .and_then(pyjson::PyNum::to_i64)
        .ok_or_else(|| {
            BenchmarkError::Unsupported(format!(
                "{}: {field} is {} {}; Rust reads only a 64-bit integer here",
                py_str(source),
                value.type_name(),
                value.repr()
            ))
        })
}

/// A JSON value a loader stores in a float `Track` field (`np.float64(value)` later).
fn json_f64(value: &pyjson::Json, field: &str, source: &Path) -> Result<f64, BenchmarkError> {
    value.number().map(pyjson::PyNum::to_f64).ok_or_else(|| {
        BenchmarkError::Unsupported(format!(
            "{}: {field} is {} {}; Rust reads only a number here",
            py_str(source),
            value.type_name(),
            value.repr()
        ))
    })
}

/// A file or directory name as a `str`; Rust cannot hold Python's surrogate-escaped names.
fn entry_name(name: OsString, dir: &Path) -> Result<String, BenchmarkError> {
    name.into_string().map_err(|name| {
        BenchmarkError::Unsupported(format!(
            "{}: entry {:?} is not valid UTF-8",
            py_str(dir),
            name
        ))
    })
}

/// `sorted(p.name for p in dir.iterdir() if p.is_dir())`.
fn sorted_subdirs(dir: &Path) -> Result<Vec<String>, BenchmarkError> {
    let dir = py_path(dir);
    let entries = fs::read_dir(&dir).map_err(|err| BenchmarkError::os(&err, &dir))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| BenchmarkError::os(&err, &dir))?;
        if entry.path().is_dir() {
            names.push(entry_name(entry.file_name(), &dir)?);
        }
    }
    names.sort();
    Ok(names)
}

/// The names `dir.glob("*{suffix}")` yields: every entry (file, directory, dot file)
/// whose name ends with `suffix`. Like pathlib, an unreadable directory yields nothing.
fn glob_suffix(dir: &Path, suffix: &str) -> Result<Vec<String>, BenchmarkError> {
    let dir = py_path(dir);
    let Ok(entries) = fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    let mut names = Vec::new();
    for entry in entries.flatten() {
        let name = entry_name(entry.file_name(), &dir)?;
        if name.ends_with(suffix) {
            names.push(name);
        }
    }
    Ok(names)
}

/// `default_data_root()` given the value of `MOTEVAL_DATA_ROOT`.
fn data_root_from(value: Option<OsString>) -> Result<PathBuf, BenchmarkError> {
    match value {
        None => Ok(PathBuf::from("data/benchmarks")),
        Some(raw) if pytext::strip(&raw.to_string_lossy()).is_empty() => {
            Err(BenchmarkError::EmptyDataRoot)
        }
        Some(raw) => Ok(py_path(Path::new(&raw))),
    }
}

/// `default_data_root()`: `MOTEVAL_DATA_ROOT` when set (an empty or blank value is an
/// error), else `data/benchmarks` relative to the working directory.
pub fn default_data_root() -> Result<PathBuf, BenchmarkError> {
    data_root_from(std::env::var_os("MOTEVAL_DATA_ROOT"))
}

/// `load_dataset(name, root, split)`: a built-in benchmark by name. `root` defaults to
/// `default_data_root()/<name>` and `split` to the loader's default.
pub fn load_dataset(
    name: &str,
    root: Option<&Path>,
    split: Option<&str>,
) -> Result<MOTDataset, BenchmarkError> {
    load_dataset_with_env(name, root, split, std::env::var_os("MOTEVAL_DATA_ROOT"))
}

/// `load_dataset` with the value of `MOTEVAL_DATA_ROOT` passed in (Rust tests cannot set
/// the environment without `unsafe`).
#[doc(hidden)]
pub fn load_dataset_with_env(
    name: &str,
    root: Option<&Path>,
    split: Option<&str>,
    data_root: Option<OsString>,
) -> Result<MOTDataset, BenchmarkError> {
    let Some(&(_, loader, default_split)) = BENCHMARKS.iter().find(|(n, _, _)| *n == name) else {
        return Err(BenchmarkError::UnknownDataset(name.to_owned()));
    };
    let root = match root {
        Some(root) => py_path(root),
        None => py_join(&data_root_from(data_root)?, name),
    };
    loader(&root, split.unwrap_or(default_split))
}
