//! `json.loads` as CPython 3.14.8 runs it (the C scanner in `Modules/_json.c`), and the
//! Python operations the JSON loaders (PanAf500, ChimpACT) apply to the decoded values.
//!
//! Every error is the text Python prints: `JSONDecodeError` (`msg: line L column C (char
//! P)`, positions in code points), `KeyError` (the key's repr), `TypeError`, and
//! `ValueError` for unpacking. Values outside what a Rust value can hold are errors with
//! Rust's own text: an integer beyond `i128`, a lone UTF-16 surrogate, and nesting
//! deeper than `MAX_DEPTH`.

use super::{BenchmarkError, PyException};
use crate::formats::pytext::{IntError, int_error_message, py_float};
use crate::pyfmt::repr_str;

/// A decoded JSON value, with Python's `int`/`float` distinction kept.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    /// Pairs in document order; a repeated key's last value wins on lookup, as in `dict`.
    Object(Vec<(String, Json)>),
}

/// Nesting depth past which decoding stops (Python stops at its C recursion limit).
const MAX_DEPTH: usize = 2000;

/// Why `loads` failed, with the text Python prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JsonError {
    /// `json.JSONDecodeError`.
    Decode(String),
    /// `ValueError`: an integer literal of more than 4300 digits.
    TooManyDigits(String),
    /// A document Python decodes but Rust cannot hold (Rust's own text).
    Unsupported(String),
}

/// A failed scan: Python's `StopIteration(idx)` ("Expecting value") or a raised error.
enum Fail {
    Stop(usize),
    Raise(JsonError),
}

struct Scanner<'a> {
    s: &'a [char],
    depth: usize,
    /// The first value Python decodes but Rust cannot hold (a lone surrogate, an integer
    /// beyond `i128`), with Rust's message. A placeholder takes its place and `loads`
    /// reports it only if the whole document decodes, so every error Python raises
    /// still comes first.
    unsupported: std::cell::RefCell<Option<String>>,
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// `JSONDecodeError(msg, doc, pos)` as text.
fn decode_error(s: &[char], msg: &str, pos: usize) -> String {
    let before = &s[..pos.min(s.len())];
    let lineno = before.iter().filter(|&&c| c == '\n').count() + 1;
    let colno = match before.iter().rposition(|&c| c == '\n') {
        Some(nl) => pos - nl,
        None => pos + 1,
    };
    format!("{msg}: line {lineno} column {colno} (char {pos})")
}

impl Scanner<'_> {
    /// Record the first value Rust cannot hold (see `unsupported`).
    fn defer(&self, message: String) {
        self.unsupported.borrow_mut().get_or_insert(message);
    }

    fn err(&self, msg: &str, pos: usize) -> Fail {
        Fail::Raise(JsonError::Decode(decode_error(self.s, msg, pos)))
    }

    fn skip_ws(&self, mut idx: usize) -> usize {
        while idx < self.s.len() && is_ws(self.s[idx]) {
            idx += 1;
        }
        idx
    }

    fn matches(&self, idx: usize, word: &str) -> bool {
        word.chars()
            .enumerate()
            .all(|(k, c)| self.s.get(idx + k) == Some(&c))
    }

    /// `scan_once_unicode`.
    fn scan_once(&mut self, idx: usize) -> Result<(Json, usize), Fail> {
        let len = self.s.len();
        if idx >= len {
            return Err(Fail::Stop(idx));
        }
        match self.s[idx] {
            '"' => {
                return self
                    .scan_string(idx + 1)
                    .map(|(s, end)| (Json::Str(s), end));
            }
            '{' | '[' => {
                if self.depth >= MAX_DEPTH {
                    return Err(Fail::Raise(JsonError::Unsupported(format!(
                        "JSON nesting deeper than {MAX_DEPTH} levels"
                    ))));
                }
                self.depth += 1;
                let result = if self.s[idx] == '{' {
                    self.parse_object(idx + 1)
                } else {
                    self.parse_array(idx + 1)
                };
                self.depth -= 1;
                return result;
            }
            'n' if idx + 3 < len && self.matches(idx + 1, "ull") => {
                return Ok((Json::Null, idx + 4));
            }
            't' if idx + 3 < len && self.matches(idx + 1, "rue") => {
                return Ok((Json::Bool(true), idx + 4));
            }
            'f' if idx + 4 < len && self.matches(idx + 1, "alse") => {
                return Ok((Json::Bool(false), idx + 5));
            }
            'N' if idx + 2 < len && self.matches(idx + 1, "aN") => {
                return Ok((Json::Float(f64::NAN.abs()), idx + 3));
            }
            'I' if idx + 7 < len && self.matches(idx + 1, "nfinity") => {
                return Ok((Json::Float(f64::INFINITY), idx + 8));
            }
            '-' if idx + 8 < len && self.matches(idx + 1, "Infinity") => {
                return Ok((Json::Float(f64::NEG_INFINITY), idx + 9));
            }
            _ => {}
        }
        self.match_number(idx)
    }

    /// `_match_number_unicode`.
    fn match_number(&self, start: usize) -> Result<(Json, usize), Fail> {
        let s = self.s;
        let end_idx = s.len() - 1;
        let digit = |i: usize| s[i].is_ascii_digit();
        let mut idx = start;
        if s[idx] == '-' {
            idx += 1;
            if idx > end_idx {
                return Err(Fail::Stop(start));
            }
        }
        if ('1'..='9').contains(&s[idx]) {
            idx += 1;
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
        } else if s[idx] == '0' {
            idx += 1;
        } else {
            return Err(Fail::Stop(start));
        }
        let mut is_float = false;
        if idx < end_idx && s[idx] == '.' && digit(idx + 1) {
            is_float = true;
            idx += 2;
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
        }
        if idx < end_idx && (s[idx] == 'e' || s[idx] == 'E') {
            let e_start = idx;
            idx += 1;
            if idx < end_idx && (s[idx] == '-' || s[idx] == '+') {
                idx += 1;
            }
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
            if digit(idx - 1) {
                is_float = true;
            } else {
                idx = e_start;
            }
        }
        let numstr: String = s[start..idx].iter().collect();
        if is_float {
            let value = py_float(&numstr).expect("the scanner matched a float literal");
            return Ok((Json::Float(value), idx));
        }
        let digits = numstr.trim_start_matches('-').len();
        if digits > 4300 {
            return Err(Fail::Raise(JsonError::TooManyDigits(int_error_message(
                &numstr,
                IntError::TooManyDigits(digits),
            ))));
        }
        match numstr.parse::<i128>() {
            Ok(value) => Ok((Json::Int(value), idx)),
            Err(_) => {
                self.defer(format!(
                    "JSON integer {numstr} at char {start} is outside the 128-bit range"
                ));
                Ok((Json::Int(0), idx))
            }
        }
    }

    /// `scanstring_unicode`: `end` is the index after the opening quote.
    fn scan_string(&self, mut end: usize) -> Result<(String, usize), Fail> {
        let s = self.s;
        let len = s.len();
        let begin = end - 1;
        let mut out = String::new();
        loop {
            let mut next = end;
            let mut c = '\0';
            while next < len {
                c = s[next];
                if c == '"' || c == '\\' {
                    break;
                }
                if u32::from(c) <= 0x1f {
                    return Err(self.err("Invalid control character at", next));
                }
                next += 1;
            }
            if next >= len || (c != '"' && c != '\\') {
                return Err(self.err("Unterminated string starting at", begin));
            }
            out.extend(&s[end..next]);
            next += 1;
            if c == '"' {
                return Ok((out, next));
            }
            if next == len {
                return Err(self.err("Unterminated string starting at", begin));
            }
            let esc = s[next];
            let code: u32 = if esc != 'u' {
                end = next + 1;
                match esc {
                    '"' => u32::from('"'),
                    '\\' => u32::from('\\'),
                    '/' => u32::from('/'),
                    'b' => 0x08,
                    'f' => 0x0c,
                    'n' => 0x0a,
                    'r' => 0x0d,
                    't' => 0x09,
                    _ => return Err(self.err("Invalid \\escape", end - 2)),
                }
            } else {
                next += 1;
                end = next + 4;
                // Four hex digits that end the document decode; "Unterminated" follows.
                if end > len {
                    return Err(self.err("Invalid \\uXXXX escape", next - 1));
                }
                let hex = |from: usize| -> Option<u32> {
                    (from..from + 4).try_fold(0u32, |acc, k| Some(acc << 4 | s[k].to_digit(16)?))
                };
                let mut value =
                    hex(next).ok_or_else(|| self.err("Invalid \\uXXXX escape", end - 5))?;
                if (0xd800..=0xdbff).contains(&value)
                    && end + 6 < len
                    && s[end] == '\\'
                    && s[end + 1] == 'u'
                {
                    let low =
                        hex(end + 2).ok_or_else(|| self.err("Invalid \\uXXXX escape", end + 1))?;
                    if (0xdc00..=0xdfff).contains(&low) {
                        value = 0x10000 + (((value - 0xd800) << 10) | (low - 0xdc00));
                        end += 6;
                    }
                }
                value
            };
            match char::from_u32(code) {
                Some(ch) => out.push(ch),
                None => {
                    self.defer(format!(
                        "lone surrogate \\u{code:04x} in the JSON string starting at char \
                         {begin}; Rust strings cannot hold it"
                    ));
                    out.push(char::REPLACEMENT_CHARACTER);
                }
            }
        }
    }

    /// `_parse_object_unicode`: `idx` is the index after `{`.
    fn parse_object(&mut self, idx: usize) -> Result<(Json, usize), Fail> {
        let len = self.s.len();
        let mut pairs: Vec<(String, Json)> = Vec::new();
        let mut idx = self.skip_ws(idx);
        if idx >= len || self.s[idx] != '}' {
            loop {
                if idx >= len || self.s[idx] != '"' {
                    return Err(self.err("Expecting property name enclosed in double quotes", idx));
                }
                let (key, next) = self.scan_string(idx + 1)?;
                idx = self.skip_ws(next);
                if idx >= len || self.s[idx] != ':' {
                    return Err(self.err("Expecting ':' delimiter", idx));
                }
                idx = self.skip_ws(idx + 1);
                let (value, next) = self.scan_once(idx)?;
                pairs.push((key, value));
                idx = self.skip_ws(next);
                if idx < len && self.s[idx] == '}' {
                    break;
                }
                if idx >= len || self.s[idx] != ',' {
                    return Err(self.err("Expecting ',' delimiter", idx));
                }
                let comma = idx;
                idx = self.skip_ws(idx + 1);
                if idx < len && self.s[idx] == '}' {
                    return Err(self.err("Illegal trailing comma before end of object", comma));
                }
            }
        }
        Ok((Json::Object(pairs), idx + 1))
    }

    /// `_parse_array_unicode`: `idx` is the index after `[`.
    fn parse_array(&mut self, idx: usize) -> Result<(Json, usize), Fail> {
        let len = self.s.len();
        let mut values = Vec::new();
        let mut idx = self.skip_ws(idx);
        if idx >= len || self.s[idx] != ']' {
            loop {
                let (value, next) = self.scan_once(idx)?;
                values.push(value);
                idx = self.skip_ws(next);
                if idx < len && self.s[idx] == ']' {
                    break;
                }
                if idx >= len || self.s[idx] != ',' {
                    return Err(self.err("Expecting ',' delimiter", idx));
                }
                let comma = idx;
                idx = self.skip_ws(idx + 1);
                if idx < len && self.s[idx] == ']' {
                    return Err(self.err("Illegal trailing comma before end of array", comma));
                }
            }
        }
        Ok((Json::Array(values), idx + 1))
    }
}

/// `json.loads(text)` for a `str`.
pub fn loads(text: &str) -> Result<Json, JsonError> {
    let chars: Vec<char> = text.chars().collect();
    let decode = |msg: &str, pos: usize| JsonError::Decode(decode_error(&chars, msg, pos));
    if chars.first() == Some(&'\u{feff}') {
        return Err(decode("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0));
    }
    let mut scanner = Scanner {
        s: &chars,
        depth: 0,
        unsupported: std::cell::RefCell::new(None),
    };
    let start = scanner.skip_ws(0);
    let (value, end) = match scanner.scan_once(start) {
        Ok(done) => done,
        Err(Fail::Stop(pos)) => return Err(decode("Expecting value", pos)),
        Err(Fail::Raise(err)) => return Err(err),
    };
    let end = scanner.skip_ws(end);
    if end != chars.len() {
        return Err(decode("Extra data", end));
    }
    if let Some(message) = scanner.unsupported.into_inner() {
        return Err(JsonError::Unsupported(message));
    }
    Ok(value)
}

/// Python's `repr(float)`: shortest round-trip digits, exponent form when the decimal
/// exponent is below -4 or at least 16.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    let decpt = exp + 1;
    let ndigits = digits.len() as i32;
    let body = if decpt <= -4 || decpt > 16 {
        let (first, rest) = digits.split_at(1);
        let dot = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{first}{dot}e{esign}{:02}", exp.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= ndigits {
        format!("{digits}{}.0", "0".repeat((decpt - ndigits) as usize))
    } else {
        let (int_part, frac) = digits.split_at(decpt as usize);
        format!("{int_part}.{frac}")
    };
    format!("{sign}{body}")
}

/// A value usable as a `dict` key, compared as Python compares keys.
///
/// `1`, `1.0` and `True` are one key; a NaN never equals another NaN object.
#[derive(Clone, Debug, PartialEq)]
pub enum PyKey {
    None,
    Int(i128),
    /// A float that is not equal to any integer.
    Float(f64),
    Str(String),
}

impl Json {
    /// `type(x).__name__`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "NoneType",
            Json::Bool(_) => "bool",
            Json::Int(_) => "int",
            Json::Float(_) => "float",
            Json::Str(_) => "str",
            Json::Array(_) => "list",
            Json::Object(_) => "dict",
        }
    }

    /// `repr(x)` for a hashable value (the text of a `KeyError`).
    pub fn repr(&self) -> String {
        match self {
            Json::Null => "None".to_owned(),
            Json::Bool(b) => if *b { "True" } else { "False" }.to_owned(),
            Json::Int(i) => i.to_string(),
            Json::Float(f) => float_repr(*f),
            Json::Str(s) => repr_str(s),
            Json::Array(_) | Json::Object(_) => format!("<{}>", self.type_name()),
        }
    }

    /// `x[key]` with a `str` key: Python's `KeyError` or `TypeError`.
    pub fn get(&self, key: &str) -> Result<&Json, BenchmarkError> {
        match self {
            Json::Object(pairs) => pairs
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .ok_or_else(|| BenchmarkError::python(PyException::KeyError, repr_str(key))),
            Json::Array(_) => Err(type_error(
                "list indices must be integers or slices, not str".to_owned(),
            )),
            Json::Str(_) => Err(type_error(
                "string indices must be integers, not 'str'".to_owned(),
            )),
            other => Err(type_error(format!(
                "'{}' object is not subscriptable",
                other.type_name()
            ))),
        }
    }

    /// `iter(x)`: a list's items, a dict's keys, a string's characters.
    pub fn iter(&self) -> Result<Vec<Json>, BenchmarkError> {
        match self {
            Json::Array(items) => Ok(items.clone()),
            Json::Object(pairs) => {
                // A dict iterates its distinct keys in first-insertion order.
                let mut keys: Vec<&String> = Vec::new();
                for (k, _) in pairs {
                    if !keys.contains(&k) {
                        keys.push(k);
                    }
                }
                Ok(keys.into_iter().map(|k| Json::Str(k.clone())).collect())
            }
            Json::Str(s) => Ok(s.chars().map(|c| Json::Str(c.to_string())).collect()),
            other => Err(type_error(format!(
                "'{}' object is not iterable",
                other.type_name()
            ))),
        }
    }

    /// `bool(x)`.
    pub fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Int(i) => *i != 0,
            Json::Float(f) => *f != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::Array(items) => !items.is_empty(),
            Json::Object(pairs) => !pairs.is_empty(),
        }
    }

    /// `a, b, c, d = x`.
    pub fn unpack4(&self) -> Result<[Json; 4], BenchmarkError> {
        let items = match self {
            Json::Array(_) | Json::Object(_) | Json::Str(_) => self.iter()?,
            other => {
                return Err(type_error(format!(
                    "cannot unpack non-iterable {} object",
                    other.type_name()
                )));
            }
        };
        let value_error =
            |message: String| BenchmarkError::python(PyException::ValueError, message);
        match <[Json; 4]>::try_from(items) {
            Ok(four) => Ok(four),
            Err(items) if items.len() < 4 => Err(value_error(format!(
                "not enough values to unpack (expected 4, got {})",
                items.len()
            ))),
            // Python counts the extra values of a sized container, but not of a string.
            Err(items) => Err(value_error(match self {
                Json::Str(_) => "too many values to unpack (expected 4)".to_owned(),
                _ => format!(
                    "too many values to unpack (expected 4, got {})",
                    items.len()
                ),
            })),
        }
    }

    /// `hash(x)`-compatible key for a `dict`, or Python's `TypeError`.
    pub fn key(&self) -> Result<PyKey, BenchmarkError> {
        Ok(match self {
            Json::Null => PyKey::None,
            Json::Bool(b) => PyKey::Int(i128::from(*b)),
            Json::Int(i) => PyKey::Int(*i),
            Json::Float(f) => float_key(*f),
            Json::Str(s) => PyKey::Str(s.clone()),
            Json::Array(_) | Json::Object(_) => {
                let name = self.type_name();
                return Err(type_error(format!(
                    "cannot use '{name}' as a dict key (unhashable type: '{name}')"
                )));
            }
        })
    }

    /// `x == n` for an `int` `n`.
    pub fn eq_int(&self, n: i128) -> bool {
        match self {
            Json::Bool(b) => i128::from(*b) == n,
            Json::Int(i) => *i == n,
            Json::Float(f) => float_key(*f) == PyKey::Int(n),
            _ => false,
        }
    }

    /// The value as a Python number (`bool` is an `int`), if it is one.
    pub fn number(&self) -> Option<PyNum> {
        match self {
            Json::Bool(b) => Some(PyNum::Int(i128::from(*b))),
            Json::Int(i) => Some(PyNum::Int(*i)),
            Json::Float(f) => Some(PyNum::Float(*f)),
            _ => None,
        }
    }

    /// `a - b` (Python numeric semantics: `int - int` is exact).
    pub fn sub(&self, other: &Json) -> Result<PyNum, BenchmarkError> {
        match (self.number(), other.number()) {
            (Some(PyNum::Int(a)), Some(PyNum::Int(b))) => {
                a.checked_sub(b).map(PyNum::Int).ok_or_else(|| {
                    BenchmarkError::Unsupported(format!(
                        "integer subtraction {a} - {b} is outside the 128-bit range"
                    ))
                })
            }
            (Some(a), Some(b)) => Ok(PyNum::Float(a.to_f64() - b.to_f64())),
            _ => Err(type_error(format!(
                "unsupported operand type(s) for -: '{}' and '{}'",
                self.type_name(),
                other.type_name()
            ))),
        }
    }

    /// `x * f` for a `float` `f`.
    pub fn mul_float(&self, f: f64) -> Result<f64, BenchmarkError> {
        match (self, self.number()) {
            (_, Some(n)) => Ok(n.to_f64() * f),
            (Json::Str(_) | Json::Array(_), None) => Err(type_error(
                "can't multiply sequence by non-int of type 'float'".to_owned(),
            )),
            (other, None) => Err(type_error(format!(
                "unsupported operand type(s) for *: '{}' and 'float'",
                other.type_name()
            ))),
        }
    }
}

fn type_error(message: String) -> BenchmarkError {
    BenchmarkError::python(PyException::TypeError, message)
}

/// A float key: integral floats equal the integer of the same value.
fn float_key(f: f64) -> PyKey {
    if f.fract() == 0.0 && f.abs() < 1.7e38 {
        PyKey::Int(f as i128)
    } else {
        PyKey::Float(f)
    }
}

/// A Python `int` or `float`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyNum {
    Int(i128),
    Float(f64),
}

impl PyNum {
    /// `float(x)`: an `int` rounds to the nearest double, ties to even.
    pub fn to_f64(self) -> f64 {
        match self {
            PyNum::Int(i) => i as f64,
            PyNum::Float(f) => f,
        }
    }

    /// The value as an `i64` when it is an `int` that fits.
    pub fn to_i64(self) -> Option<i64> {
        match self {
            PyNum::Int(i) => i64::try_from(i).ok(),
            PyNum::Float(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::pytext_corpus;
    use serde_json::Value;

    /// The decoded value as `dict(pairs)` would hold it, in `dump_pytext.tag`'s form.
    fn tag(value: &Json) -> Value {
        match value {
            Json::Null => serde_json::json!({"t": "null"}),
            Json::Bool(b) => serde_json::json!({"t": "bool", "v": b}),
            Json::Int(i) => serde_json::json!({"t": "int", "v": i.to_string()}),
            Json::Float(f) => serde_json::json!({"t": "float", "v": f.to_bits()}),
            Json::Str(s) => serde_json::json!({"t": "str", "v": s}),
            Json::Array(items) => {
                serde_json::json!({"t": "list", "v": items.iter().map(tag).collect::<Vec<_>>()})
            }
            Json::Object(pairs) => {
                // dict(pairs): first position, last value.
                let mut merged: Vec<(&String, &Json)> = Vec::new();
                for (k, v) in pairs {
                    match merged.iter_mut().find(|(key, _)| *key == k) {
                        Some(slot) => slot.1 = v,
                        None => merged.push((k, v)),
                    }
                }
                let items: Vec<Value> = merged
                    .into_iter()
                    .map(|(k, v)| serde_json::json!([k, tag(v)]))
                    .collect();
                serde_json::json!({"t": "dict", "v": items})
            }
        }
    }

    #[test]
    fn loads_matches_cpython() {
        let cases = pytext_corpus("json");
        assert!(cases.len() > 1200);
        for case in &cases {
            let input = case["input"].as_str().expect("input");
            let got = loads(input);
            if case.get("unsupported").is_some() {
                assert!(
                    matches!(got, Err(JsonError::Unsupported(_))),
                    "{input:?}: {got:?}"
                );
            } else if let Some(expected) = case.get("ok") {
                let value = got.unwrap_or_else(|err| panic!("{input:?}: {err:?}"));
                assert_eq!(&tag(&value), expected, "{input:?}");
            } else {
                let message = case["message"].as_str().expect("message");
                let (kind, got_message) = match got {
                    Ok(value) => panic!("{input:?}: Python raises {message}, got {value:?}"),
                    Err(JsonError::Decode(m)) => ("JSONDecodeError", m),
                    Err(JsonError::TooManyDigits(m)) => ("ValueError", m),
                    Err(JsonError::Unsupported(m)) => ("Unsupported", m),
                };
                assert_eq!(kind, case["error"], "{input:?}: {got_message}");
                assert_eq!(got_message, message, "{input:?}");
            }
        }
    }

    #[test]
    fn float_repr_matches_python() {
        for case in &pytext_corpus("format") {
            let x = f64::from_bits(case["bits"].as_u64().expect("bits"));
            assert_eq!(float_repr(x), case["repr"], "repr({x:e})");
        }
    }
}
