//! CPython 3.14.8 text semantics that the readers and loaders copy.
//!
//! The annotation readers must accept and reject exactly the strings Python does, and the
//! loaders must find the same files and print the same messages. Each function here names
//! the CPython routine it follows. The Unicode classes come from `pytables`, generated
//! from the reference interpreter.

use std::path::{Component, Path, PathBuf};

use super::pytables::{DECIMAL, LINE_BREAK, SPACE};

/// `str.isspace()` for one character (`Py_UNICODE_ISSPACE`; also `re`'s `\s`).
pub fn is_space(ch: char) -> bool {
    SPACE.binary_search(&u32::from(ch)).is_ok()
}

/// `Py_ISSPACE`: the six ASCII whitespace bytes `int()` and `float()` strip.
fn is_ascii_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// `str.strip()` with no argument.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `str.split()` with no argument: runs of whitespace separate fields.
pub fn split_whitespace(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|part| !part.is_empty()).collect()
}

/// `str.splitlines()` with no argument. `\r\n` is one boundary.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        if LINE_BREAK.binary_search(&u32::from(ch)).is_err() {
            continue;
        }
        lines.push(&s[start..i]);
        let mut end = i + ch.len_utf8();
        if ch == '\r' && chars.peek().is_some_and(|&(_, next)| next == '\n') {
            chars.next();
            end += 1;
        }
        start = end;
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// The decimal value of a Unicode decimal digit (`Py_UNICODE_TODECIMAL`).
fn decimal_value(ch: char) -> Option<u8> {
    let code = u32::from(ch);
    let idx = DECIMAL.partition_point(|&(_, hi, _)| hi < code);
    let &(lo, _, first) = DECIMAL.get(idx)?;
    (lo <= code).then(|| (first + code - lo) as u8)
}

/// `_PyUnicode_TransformDecimalAndSpaceToASCII`, then the NUL check every caller makes.
///
/// Unicode whitespace becomes a space and Unicode decimal digits become ASCII digits. Any
/// other non-ASCII character, and any NUL, makes the string unparsable (`None`).
fn to_ascii_numeric(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let code = u32::from(ch);
        if code == 0 {
            return None;
        }
        if code < 127 {
            out.push(code as u8);
        } else if is_space(ch) {
            out.push(b' ');
        } else {
            out.push(b'0' + decimal_value(ch)?);
        }
    }
    Some(out)
}

/// `_Py_string_to_number_with_underscores`: drop the underscores `float()` allows.
///
/// An underscore must follow a digit and precede a digit.
fn remove_float_underscores(s: &[u8]) -> Option<Vec<u8>> {
    if !s.contains(&b'_') {
        return Some(s.to_vec());
    }
    let mut out = Vec::with_capacity(s.len());
    let mut prev = 0u8;
    for &b in s {
        if b == b'_' {
            if !prev.is_ascii_digit() {
                return None;
            }
        } else {
            out.push(b);
            if prev == b'_' && !b.is_ascii_digit() {
                return None;
            }
        }
        prev = b;
    }
    (prev != b'_').then_some(out)
}

/// Python's `float(s)` for a `str` (`PyFloat_FromString`): `None` where Python raises.
///
/// Accepts surrounding whitespace, `_` between digits, Unicode decimal digits, and the
/// case-insensitive spellings `inf`, `infinity` and `nan` with an optional sign (`-nan`
/// has its sign bit set). Overflow gives an infinity, as in Python.
pub fn py_float(s: &str) -> Option<f64> {
    let ascii = to_ascii_numeric(s)?;
    let bytes = remove_float_underscores(&ascii)?;
    // float_from_string_inner: strip ASCII whitespace from both ends.
    let start = bytes.iter().position(|&b| !is_ascii_space(b))?;
    let end = bytes.iter().rposition(|&b| !is_ascii_space(b))? + 1;
    let body = &bytes[start..end];
    parse_decimal(body).or_else(|| parse_inf_or_nan(body))
}

/// `_Py_dg_strtod` over the whole of `body`: `[+-]? (d+ (. d*)? | . d+) ([eE] [+-]? d+)?`.
fn parse_decimal(body: &[u8]) -> Option<f64> {
    let mut i = 0;
    if matches!(body.first(), Some(b'+' | b'-')) {
        i += 1;
    }
    let int_digits = body[i..].iter().take_while(|b| b.is_ascii_digit()).count();
    i += int_digits;
    let mut frac_digits = 0;
    if body.get(i) == Some(&b'.') {
        frac_digits = body[i + 1..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
        i += 1 + frac_digits;
    }
    if int_digits + frac_digits == 0 {
        return None;
    }
    if matches!(body.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(body.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let exp_digits = body[j..].iter().take_while(|b| b.is_ascii_digit()).count();
        if exp_digits > 0 {
            i = j + exp_digits;
        }
    }
    if i != body.len() {
        return None;
    }
    // The syntax is checked; Rust's parser is correctly rounded like dtoa and reads this
    // grammar (an exponent of any length saturates to an infinity or a zero).
    std::str::from_utf8(body).ok()?.parse::<f64>().ok()
}

/// `_Py_parse_inf_or_nan` over the whole of `body`.
fn parse_inf_or_nan(body: &[u8]) -> Option<f64> {
    let (negate, rest) = match body.first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    let lower = rest.to_ascii_lowercase();
    let value = match lower.as_slice() {
        b"inf" | b"infinity" => f64::INFINITY,
        b"nan" => f64::NAN.abs(),
        _ => return None,
    };
    Some(if negate { -value } else { value })
}

/// The outcome of Python's `int(s)` for a `str`, base 10.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyInt {
    /// Python returns this value.
    Value(i64),
    /// Python returns an integer that `i64` cannot hold; `negative` gives its sign.
    OutOfRange { negative: bool },
}

/// Why Python's `int(s)` raises `ValueError`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntError {
    /// `invalid literal for int() with base 10: ...`.
    Invalid,
    /// More digits than `sys.get_int_max_str_digits()` allows; holds the digit count.
    TooManyDigits(usize),
}

/// CPython's default `sys.get_int_max_str_digits()`: longer digit strings raise.
const MAX_STR_DIGITS: usize = 4300;

/// `str(ValueError)` for `int(s)` failing with `err`.
pub fn int_error_message(s: &str, err: IntError) -> String {
    match err {
        // `%.200R`: the repr, cut to its first 200 characters.
        IntError::Invalid => format!(
            "invalid literal for int() with base 10: {}",
            crate::pyfmt::repr_str(s)
                .chars()
                .take(200)
                .collect::<String>()
        ),
        IntError::TooManyDigits(digits) => format!(
            "Exceeds the limit ({MAX_STR_DIGITS} digits) for integer string conversion: \
             value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
        ),
    }
}

/// Python's `int(s)` for a `str` (`PyLong_FromUnicodeObject`, base 10).
///
/// Accepts surrounding whitespace, one sign, `_` between digits, and Unicode decimal
/// digits. More than 4300 digits raise, as in Python.
pub fn py_int(s: &str) -> Result<PyInt, IntError> {
    let ascii = to_ascii_numeric(s).ok_or(IntError::Invalid)?;
    let mut i = ascii.iter().take_while(|&&b| is_ascii_space(b)).count();
    let mut negative = false;
    match ascii.get(i) {
        Some(b'+') => i += 1,
        Some(b'-') => {
            negative = true;
            i += 1;
        }
        _ => {}
    }
    // long_from_string_base: digits with single underscores between them.
    if ascii.get(i) == Some(&b'_') {
        return Err(IntError::Invalid);
    }
    let mut digits = Vec::new();
    let mut prev = 0u8;
    while let Some(&b) = ascii.get(i) {
        if b == b'_' {
            if prev == b'_' {
                return Err(IntError::Invalid);
            }
        } else if b.is_ascii_digit() {
            digits.push(b - b'0');
        } else {
            break;
        }
        prev = b;
        i += 1;
    }
    if prev == b'_' || digits.is_empty() {
        return Err(IntError::Invalid);
    }
    if !ascii[i..].iter().all(|&b| is_ascii_space(b)) {
        return Err(IntError::Invalid);
    }
    if digits.len() > MAX_STR_DIGITS {
        return Err(IntError::TooManyDigits(digits.len()));
    }
    let mut magnitude: u64 = 0;
    for &d in &digits {
        match magnitude
            .checked_mul(10)
            .and_then(|m| m.checked_add(u64::from(d)))
        {
            Some(m) => magnitude = m,
            None => return Ok(PyInt::OutOfRange { negative }),
        }
    }
    let value = if negative {
        0i64.checked_sub_unsigned(magnitude)
    } else {
        i64::try_from(magnitude).ok()
    };
    Ok(value.map_or(PyInt::OutOfRange { negative }, PyInt::Value))
}

/// `str(pathlib.PurePosixPath(p))`: the path as pathlib normalizes it.
///
/// Repeated slashes and `.` parts collapse, a trailing slash goes, exactly two leading
/// slashes stay, `..` stays, and an empty path is `.`. Paths built through this function
/// also behave like pathlib's on disk (`Path("file/").is_file()` is true in Python).
pub fn py_path(p: &Path) -> PathBuf {
    let raw = p.as_os_str().as_encoded_bytes();
    let two_slash_root = raw.starts_with(b"//") && !raw.starts_with(b"///");
    let mut out = PathBuf::new();
    for component in p.components() {
        match component {
            Component::RootDir => out.push(if two_slash_root { "//" } else { "/" }),
            Component::CurDir => {}
            Component::ParentDir => out.push(".."),
            Component::Normal(part) => out.push(part),
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// `base / part` in pathlib: join, then normalize.
pub fn py_join(base: &Path, part: impl AsRef<Path>) -> PathBuf {
    py_path(&base.join(part))
}

/// `str(path)` for a message.
pub fn py_str(p: &Path) -> String {
    py_path(p).to_string_lossy().into_owned()
}

/// `PurePath.name`: the last part, or `""` for a root or `.`.
pub fn py_name(p: &Path) -> String {
    let normalized = py_path(p);
    match normalized.components().next_back() {
        Some(Component::Normal(part)) => part.to_string_lossy().into_owned(),
        Some(Component::ParentDir) => "..".to_owned(),
        _ => String::new(),
    }
}

/// `PurePath.stem` of a file name (Python 3.14): the name without its last suffix,
/// unless that leaves nothing but dots.
pub fn py_stem(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) if !name[..i].trim_start_matches('.').is_empty() => &name[..i],
        _ => name,
    }
}

/// `str(OSError)` as Python prints it for a call on `path`:
/// `[Errno 2] No such file or directory: '/x'`.
pub fn os_error_message(err: &std::io::Error, path: &Path) -> String {
    let filename = crate::pyfmt::repr_str(&py_str(path));
    match err.raw_os_error() {
        Some(code) => {
            // Rust prints `<strerror text> (os error N)`; Python prints the same strerror.
            let text = err.to_string();
            let suffix = format!(" (os error {code})");
            let strerror = text.strip_suffix(&suffix).unwrap_or(&text);
            format!("[Errno {code}] {strerror}: {filename}")
        }
        None => format!("{err}: {filename}"),
    }
}

/// Python's `UnicodeDecodeError` text for the first invalid UTF-8 in `data`, decoded in
/// one call with `final=True` (as `Path.read_text` decodes), or `None` for valid UTF-8.
///
/// Rust reports the maximal invalid subpart, which is the range CPython's decoder reports.
pub fn utf8_error_message(data: &[u8]) -> Option<String> {
    let err = std::str::from_utf8(data).err()?;
    let start = err.valid_up_to();
    let (end, reason) = match err.error_len() {
        None => (data.len(), "unexpected end of data"),
        Some(len) => {
            let lead = data[start];
            let reason = if (0xc2..=0xf4).contains(&lead) {
                "invalid continuation byte"
            } else {
                "invalid start byte"
            };
            (start + len, reason)
        }
    };
    Some(if end - start == 1 {
        format!(
            "'utf-8' codec can't decode byte 0x{:02x} in position {start}: {reason}",
            data[start]
        )
    } else {
        format!(
            "'utf-8' codec can't decode bytes in position {start}-{}: {reason}",
            end - 1
        )
    })
}

/// Number of code points before byte offset `byte` of `s` (a Python string index).
pub fn char_index(s: &str, byte: usize) -> usize {
    s[..byte].chars().count()
}
