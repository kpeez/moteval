//! On-disk annotation formats, ported from `moteval/formats.py`: MOTChallenge box rows
//! (`Track`, `read_mot`, `write_mot`) and MOTS mask rows (`MaskTrack`, `read_mots`,
//! `write_mots`).
//!
//! The formats-protocol-loaders package fills the function bodies. Python's `float()`
//! accepts strings that `str::parse::<f64>` rejects or reads differently; the readers must
//! accept and reject exactly the same strings.

use std::fmt;
use std::path::Path;

/// One box at one frame, tagged with its track id.
///
/// For ground-truth rows `conf` is the file's 7th column (the "consider" flag); for
/// predictions it is the detection confidence. `class_id` is 1 unless the reader is asked
/// to read the class column.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub frame: i64,
    pub track_id: i64,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub conf: f64,
    pub class_id: i64,
}

/// One RLE mask at one frame, tagged with its track id.
///
/// `rle` is the compressed counts string exactly as read from the file. Class 10 marks
/// ignore regions in MOTS GT files; the loader routes those rows, `read_mots` returns all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskTrack {
    pub frame: i64,
    pub track_id: i64,
    pub class_id: i64,
    pub img_h: i64,
    pub img_w: i64,
    pub rle: String,
}

/// Errors from reading or writing annotation files. Messages match the Python text.
#[derive(Debug)]
pub enum FormatError {
    Io(std::io::Error),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for FormatError {}

/// Parse a MOTChallenge txt file into `Track` rows (`read_mot`).
pub fn read_mot(_path: &Path, _class_column: bool) -> Result<Vec<Track>, FormatError> {
    todo!("formats-protocol-loaders: port moteval.formats.read_mot")
}

/// Write `Track` rows as MOTChallenge predictions (`write_mot`).
pub fn write_mot(_path: &Path, _tracks: &[Track]) -> Result<(), FormatError> {
    todo!("formats-protocol-loaders: port moteval.formats.write_mot")
}

/// Parse a MOTS txt file into `MaskTrack` rows (`read_mots`).
pub fn read_mots(_path: &Path) -> Result<Vec<MaskTrack>, FormatError> {
    todo!("formats-protocol-loaders: port moteval.formats.read_mots")
}

/// Write `MaskTrack` rows as a MOTS txt file (`write_mots`).
pub fn write_mots(_path: &Path, _tracks: &[MaskTrack]) -> Result<(), FormatError> {
    todo!("formats-protocol-loaders: port moteval.formats.write_mots")
}
