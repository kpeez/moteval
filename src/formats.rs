#[rustfmt::skip]
mod pytables;
pub mod pytext;

use std::fmt;
use std::fs;
use std::path::Path;

use crate::pyfmt::repr_str;
use pytext::{PyInt, py_float, py_int, py_str};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskTrack {
    pub frame: i64,
    pub track_id: i64,
    pub class_id: i64,
    pub img_h: i64,
    pub img_w: i64,
    pub rle: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowFormat {
    Mot,
    Mots,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatError {
    MalformedRow {
        format: RowFormat,
        path: String,
        lineno: usize,
        line: String,
    },
    RowWithoutClass {
        path: String,
        lineno: usize,
        line: String,
    },
    IntegerOutOfRange {
        path: String,
        lineno: usize,
        line: String,
    },
    Os(String),
    Decode(String),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedRow {
                format,
                path,
                lineno,
                line,
            } => {
                let name = match format {
                    RowFormat::Mot => "MOT",
                    RowFormat::Mots => "MOTS",
                };
                write!(
                    f,
                    "malformed {name} row in {path}:{lineno}: {}",
                    repr_str(line)
                )
            }
            Self::RowWithoutClass { path, lineno, line } => write!(
                f,
                "MOT row without a class (column 8) in {path}:{lineno}: {}",
                repr_str(line)
            ),
            Self::IntegerOutOfRange { path, lineno, line } => write!(
                f,
                "integer outside the 64-bit range in {path}:{lineno}: {}",
                repr_str(line)
            ),
            Self::Os(message) | Self::Decode(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for FormatError {}

impl FormatError {
    pub fn python_exception(&self) -> &'static str {
        match self {
            Self::MalformedRow { .. } | Self::RowWithoutClass { .. } => "ValueError",
            Self::IntegerOutOfRange { .. } => "Unsupported",
            Self::Os(_) => "OSError",
            Self::Decode(_) => "UnicodeDecodeError",
        }
    }
}

pub fn read_text(path: &Path) -> Result<String, FormatError> {
    let path = pytext::py_path(path);
    let bytes =
        fs::read(&path).map_err(|err| FormatError::Os(pytext::os_error_message(&err, &path)))?;
    match String::from_utf8(bytes) {
        Ok(text) => Ok(text),
        Err(err) => Err(FormatError::Decode(
            pytext::utf8_error_message(err.as_bytes()).expect("invalid UTF-8 has an error"),
        )),
    }
}

fn write_text(path: &Path, text: &str) -> Result<(), FormatError> {
    let path = pytext::py_path(path);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|err| FormatError::Os(pytext::os_error_message(&err, parent)))?;
    }
    fs::write(&path, text).map_err(|err| FormatError::Os(pytext::os_error_message(&err, &path)))
}

fn int_field(s: &str) -> Option<PyInt> {
    py_int(s).ok()
}

pub fn read_mot(path: &Path, class_column: bool) -> Result<Vec<Track>, FormatError> {
    let text = read_text(path)?;
    let path_str = py_str(path);
    let mut tracks = Vec::new();
    for (index, raw_line) in pytext::splitlines(&text).into_iter().enumerate() {
        let lineno = index + 1;
        let line = pytext::strip(raw_line);
        if line.is_empty() {
            continue;
        }
        let malformed = || FormatError::MalformedRow {
            format: RowFormat::Mot,
            path: path_str.clone(),
            lineno,
            line: line.to_owned(),
        };
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < if class_column { 8 } else { 6 } {
            return Err(malformed());
        }
        let conf = if fields.len() > 6 {
            py_float(fields[6])
        } else {
            Some(1.0)
        };
        let frame = int_field(fields[0]);
        let track_id = int_field(fields[1]);
        let x = py_float(fields[2]);
        let y = py_float(fields[3]);
        let w = py_float(fields[4]);
        let h = py_float(fields[5]);
        let class_id = if class_column {
            int_field(fields[7])
        } else {
            Some(PyInt::Value(1))
        };
        let (
            Some(conf),
            Some(frame),
            Some(track_id),
            Some(x),
            Some(y),
            Some(w),
            Some(h),
            Some(class_id),
        ) = (conf, frame, track_id, x, y, w, h, class_id)
        else {
            return Err(malformed());
        };
        let class_negative = match class_id {
            PyInt::Value(v) => v < 0,
            PyInt::OutOfRange { negative } => negative,
        };
        if class_negative {
            return Err(FormatError::RowWithoutClass {
                path: path_str.clone(),
                lineno,
                line: line.to_owned(),
            });
        }
        let (PyInt::Value(frame), PyInt::Value(track_id), PyInt::Value(class_id)) =
            (frame, track_id, class_id)
        else {
            return Err(FormatError::IntegerOutOfRange {
                path: path_str.clone(),
                lineno,
                line: line.to_owned(),
            });
        };
        tracks.push(Track {
            frame,
            track_id,
            x,
            y,
            w,
            h,
            conf,
            class_id,
        });
    }
    Ok(tracks)
}

fn format_fixed(x: f64, precision: usize) -> String {
    if x.is_nan() {
        "nan".to_owned()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_owned()
    } else {
        format!("{x:.precision$}")
    }
}

pub fn write_mot(path: &Path, tracks: &[Track]) -> Result<(), FormatError> {
    let mut rows: Vec<&Track> = tracks.iter().collect();
    rows.sort_by_key(|t| (t.frame, t.track_id));
    let lines: Vec<String> = rows
        .iter()
        .map(|t| {
            format!(
                "{},{},{},{},{},{},{},{},-1,-1",
                t.frame,
                t.track_id,
                format_fixed(t.x, 2),
                format_fixed(t.y, 2),
                format_fixed(t.w, 2),
                format_fixed(t.h, 2),
                format_fixed(t.conf, 4),
                t.class_id
            )
        })
        .collect();
    write_text(path, &join_lines(&lines))
}

fn join_lines(lines: &[String]) -> String {
    let mut text = lines.join("\n");
    if !lines.is_empty() {
        text.push('\n');
    }
    text
}

pub fn read_mots(path: &Path) -> Result<Vec<MaskTrack>, FormatError> {
    let text = read_text(path)?;
    let path_str = py_str(path);
    let mut tracks = Vec::new();
    for (index, raw_line) in pytext::splitlines(&text).into_iter().enumerate() {
        let lineno = index + 1;
        let line = pytext::strip(raw_line);
        if line.is_empty() {
            continue;
        }
        let row_error = |out_of_range: bool| {
            let (path, line) = (path_str.clone(), line.to_owned());
            if out_of_range {
                FormatError::IntegerOutOfRange { path, lineno, line }
            } else {
                FormatError::MalformedRow {
                    format: RowFormat::Mots,
                    path,
                    lineno,
                    line,
                }
            }
        };
        let fields = pytext::split_whitespace(line);
        if fields.len() != 6 {
            return Err(row_error(false));
        }
        let ints: Vec<Option<PyInt>> = fields[..5].iter().map(|f| int_field(f)).collect();
        if ints.iter().any(Option::is_none) {
            return Err(row_error(false));
        }
        let mut values = [0i64; 5];
        for (slot, parsed) in values.iter_mut().zip(&ints) {
            match parsed {
                Some(PyInt::Value(v)) => *slot = *v,
                _ => return Err(row_error(true)),
            }
        }
        let [frame, track_id, class_id, img_h, img_w] = values;
        tracks.push(MaskTrack {
            frame,
            track_id,
            class_id,
            img_h,
            img_w,
            rle: fields[5].to_owned(),
        });
    }
    Ok(tracks)
}

pub fn write_mots(path: &Path, tracks: &[MaskTrack]) -> Result<(), FormatError> {
    let mut rows: Vec<&MaskTrack> = tracks.iter().collect();
    rows.sort_by_key(|t| (t.frame, t.track_id));
    let lines: Vec<String> = rows
        .iter()
        .map(|t| {
            format!(
                "{} {} {} {} {} {}",
                t.frame, t.track_id, t.class_id, t.img_h, t.img_w, t.rle
            )
        })
        .collect();
    write_text(path, &join_lines(&lines))
}

pub fn missing_prediction_files<'a>(
    pred_dir: &Path,
    names: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    names
        .into_iter()
        .filter(|name| !pytext::py_join(pred_dir, format!("{name}.txt")).is_file())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
pub(crate) fn pytext_corpus(topic: &str) -> Vec<serde_json::Value> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/integration/data/pytext")
        .join(format!("{topic}.json"));
    let text = std::fs::read_to_string(&path).expect("read the pytext corpus");
    let json: serde_json::Value = serde_json::from_str(&text).expect("parse the pytext corpus");
    json.as_array().expect("a list of cases").clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn os_errors_print_as_python_does() {
        let missing = read_text(Path::new("/nonexist/x.txt")).unwrap_err();
        assert_eq!(
            missing.to_string(),
            "[Errno 2] No such file or directory: '/nonexist/x.txt'"
        );
        assert_eq!(missing.python_exception(), "OSError");
        let dir = std::env::temp_dir();
        let is_dir = read_text(&dir).unwrap_err();
        assert_eq!(
            is_dir.to_string(),
            format!("[Errno 21] Is a directory: {}", repr_str(&py_str(&dir)))
        );
        assert_eq!(is_dir.python_exception(), "OSError");
    }

    #[test]
    fn fixed_point_formatting_matches_python() {
        let cases = pytext_corpus("format");
        assert!(cases.len() > 3000);
        for case in &cases {
            let x = f64::from_bits(case["bits"].as_u64().expect("bits"));
            assert_eq!(format_fixed(x, 2), case["f2"], "format({x:e}, '.2f')");
            assert_eq!(format_fixed(x, 4), case["f4"], "format({x:e}, '.4f')");
        }
    }
}
