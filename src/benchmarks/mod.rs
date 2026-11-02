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

pub type Loader = fn(&Path, &str) -> Result<MOTDataset, BenchmarkError>;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyException {
    OsError,
    UnicodeDecodeError,
    JsonDecodeError,
    ValueError,
    KeyError,
    TypeError,
    InterpolationError,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BenchmarkError {
    EmptyDataRoot,
    UnknownDataset(String),
    SplitDirNotFound(String),
    LabelsDirNotFound(String),
    UnknownSplit {
        benchmark: &'static str,
        split: String,
        expected: &'static str,
    },
    SplitFileNotFound {
        split: String,
        path: String,
    },
    MissingFile {
        file: &'static str,
        sequence: String,
        path: String,
    },
    MalformedSeqinfo {
        sequence: String,
        path: String,
        detail: String,
    },
    EmptyGt(String),
    Python {
        exception: PyException,
        message: String,
    },
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
    pub fn python_exception(&self) -> &'static str {
        match self {
            Self::Python { exception, .. } => match exception {
                PyException::OsError => "OSError",
                PyException::UnicodeDecodeError => "UnicodeDecodeError",
                PyException::JsonDecodeError => "JSONDecodeError",
                PyException::ValueError => "ValueError",
                PyException::KeyError => "KeyError",
                PyException::TypeError => "TypeError",
                PyException::InterpolationError => "InterpolationError",
            },
            Self::Unsupported(_) => "Unsupported",
            Self::Format(err) => err.python_exception(),
            Self::Convert(err) => err.python_exception(),
            Self::Model(ModelError::WrongGeometry { .. }) => "TypeError",
            _ => "ValueError",
        }
    }

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

fn read_text(path: &Path) -> Result<String, BenchmarkError> {
    crate::formats::read_text(path).map_err(|err| match err {
        FormatError::Decode(message) => {
            BenchmarkError::python(PyException::UnicodeDecodeError, message)
        }
        other => BenchmarkError::python(PyException::OsError, other.to_string()),
    })
}

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

fn entry_name(name: OsString, dir: &Path) -> Result<String, BenchmarkError> {
    name.into_string().map_err(|name| {
        BenchmarkError::Unsupported(format!(
            "{}: entry {:?} is not valid UTF-8",
            py_str(dir),
            name
        ))
    })
}

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

fn glob_suffix(dir: &Path, suffix: &str) -> Result<Vec<String>, BenchmarkError> {
    let dir = py_path(dir);
    // CPython `Path.glob` yields nothing for an unreadable directory.
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

#[doc(hidden)]
pub fn default_data_root_with_env(value: Option<OsString>) -> Result<PathBuf, BenchmarkError> {
    match value {
        None => Ok(PathBuf::from("data/benchmarks")),
        Some(raw) if pytext::strip(&raw.to_string_lossy()).is_empty() => {
            Err(BenchmarkError::EmptyDataRoot)
        }
        Some(raw) => Ok(py_path(Path::new(&raw))),
    }
}

pub fn default_data_root() -> Result<PathBuf, BenchmarkError> {
    default_data_root_with_env(std::env::var_os("MOTEVAL_DATA_ROOT"))
}

pub fn load_dataset(
    name: &str,
    root: Option<&Path>,
    split: Option<&str>,
) -> Result<MOTDataset, BenchmarkError> {
    load_dataset_with_env(name, root, split, std::env::var_os("MOTEVAL_DATA_ROOT"))
}

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
        None => py_join(&default_data_root_with_env(data_root)?, name),
    };
    loader(&root, split.unwrap_or(default_split))
}
