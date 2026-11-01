//! Python-compatible text output, written by hand (no crate).
//!
//! - [`repr_str`]: `repr(s)` for a `str`; error messages quote names with `{name!r}`.
//! - [`repr_float`]: `repr(x)` and `str(x)` for a `float`: the shortest digits that round
//!   trip, positional from `1e-4` up to (not including) `1e16` with `.0` on integral
//!   values, exponent form otherwise (`1e-05`, `1e+16`).
//! - [`format_g`]: `format(x, ".{p}g")`, which the CLI table uses as `"{:1.5g}"`.
//! - [`json_float`], [`json_str`]: the scalars of `json.dumps` with its defaults
//!   (`NaN`, `Infinity`, `-Infinity`; `ensure_ascii=True`).
//! - [`csv_row`]: one `csv.writer` row with its defaults (`,`, minimal quoting with `"`,
//!   `\r\n` line end).
//! - [`path_str`]: `str(pathlib.Path(s))` on POSIX, which the CLI prints in messages.
//!
//! The digit strings come from Rust's own float formatting, which gives the shortest
//! round-trip digits (`{:e}`) and correctly rounded fixed-precision digits (`{:.Ne}`), the
//! same two operations CPython's `dtoa.c` performs. `rust/moteval/tests/pyfmt.rs` checks
//! every function against strings dumped from CPython 3.14.8 by
//! `rust/scripts/dump_pyfmt.py`, on more than 100,000 seeded doubles plus edge cases.

mod printable;

/// The decimal digits and the scientific exponent of a finite, non-negative `x`:
/// `x ~ d1.d2d3... * 10^exp`. `precision` `None` gives the shortest digits that round
/// trip, as CPython's `dtoa` mode 0 picks them; `Some(p)` gives `p` correctly rounded
/// significant digits (ties to even). Trailing zeros are removed; zero gives `("0", 0)`.
fn digits_and_exponent(x: f64, precision: Option<usize>) -> (String, i32) {
    debug_assert!(x.is_finite() && x.is_sign_positive());
    match precision {
        Some(p) => split_sci(&format!("{x:.*e}", p.saturating_sub(1))),
        None => {
            // Rust's `{:e}` finds the shortest length n, but when two n-digit strings
            // round trip and lie equally close to x it rounds the tie up, where dtoa
            // rounds it to even (`-4092142886402.53125` is `...2.5312` in Python). The
            // correctly rounded n-digit string is dtoa's choice whenever it round trips.
            let (digits, exponent) = split_sci(&format!("{x:e}"));
            let nearest = format!("{x:.*e}", digits.len() - 1);
            if nearest.parse::<f64>() == Ok(x) {
                split_sci(&nearest)
            } else {
                (digits, exponent)
            }
        }
    }
}

/// Split Rust's `{:e}` text (`1.2345e-5`) into its digits without trailing zeros and its
/// exponent.
fn split_sci(text: &str) -> (String, i32) {
    let (mantissa, exponent) = text
        .split_once('e')
        .expect("Rust's {:e} output has an exponent");
    let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let exponent = exponent
        .parse()
        .expect("Rust's {:e} exponent is an integer");
    (digits, exponent)
}

/// CPython `format_float_short` layout of `digits` with the decimal point after
/// `decpt` digits: positional or exponent form, with `.0` added to positional integers
/// when `add_dot_0` is set.
fn layout(negative: bool, digits: &str, decpt: i32, use_exp: bool, add_dot_0: bool) -> String {
    let mut out = String::with_capacity(digits.len() + 8);
    if negative {
        out.push('-');
    }
    let n = digits.len() as i32;
    if use_exp {
        let exp = decpt - 1;
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.unsigned_abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', decpt.unsigned_abs() as usize));
        out.push_str(digits);
    } else if decpt < n {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    } else {
        out.push_str(digits);
        out.extend(std::iter::repeat_n('0', (decpt - n) as usize));
        if add_dot_0 {
            out.push_str(".0");
        }
    }
    out
}

/// Python's `repr(x)` (and `str(x)`) for a `float`.
pub fn repr_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let (digits, exponent) = digits_and_exponent(x.abs(), None);
    let decpt = exponent + 1;
    // CPython `float_repr_style == 'short'`: mode 0, exponent form when decpt <= -4 or
    // decpt > 16.
    layout(
        x.is_sign_negative(),
        &digits,
        decpt,
        decpt <= -4 || decpt > 16,
        true,
    )
}

/// Python's `format(x, f".{precision}g")` for a `float` (precision 0 counts as 1).
///
/// Exponent form when the rounded exponent is below -4 or at least `precision`; trailing
/// zeros and a trailing decimal point are removed; the exponent has at least two digits.
pub fn format_g(x: f64, precision: usize) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let precision = precision.max(1);
    let (digits, exponent) = digits_and_exponent(x.abs(), Some(precision));
    let decpt = exponent + 1;
    let use_exp = decpt <= -4 || i64::from(decpt) > precision as i64;
    layout(x.is_sign_negative(), &digits, decpt, use_exp, false)
}

/// How `json.dumps` writes a `float` with its defaults (`allow_nan=True`).
pub fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_string()
    } else if x == f64::INFINITY {
        "Infinity".to_string()
    } else if x == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        repr_float(x)
    }
}

/// How `json.dumps` writes a `str` with its defaults (`ensure_ascii=True`): printable
/// ASCII stays, `"` and `\` and the five short escapes are escaped, and every other
/// character is `\uXXXX` (lowercase hex, a surrogate pair above U+FFFF).
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

/// One `csv.writer` row with its defaults: fields joined by `,`, a field quoted with `"`
/// (inner quotes doubled) when it holds `,`, `"`, `\r`, or `\n`, and the row ended by
/// `\r\n`. Non-string fields are passed in already converted with `str()`.
pub fn csv_row<S: AsRef<str>>(fields: &[S]) -> String {
    let mut out = String::new();
    for (index, field) in fields.iter().enumerate() {
        let field = field.as_ref();
        if index > 0 {
            out.push(',');
        }
        // A row of one empty field is written `""`, so it reads back as a row.
        let quote =
            field.contains([',', '"', '\r', '\n']) || (fields.len() == 1 && field.is_empty());
        if quote {
            out.push('"');
            out.push_str(&field.replace('"', "\"\""));
            out.push('"');
        } else {
            out.push_str(field);
        }
    }
    out.push_str("\r\n");
    out
}

/// `str(pathlib.PurePosixPath(s))`: repeated slashes collapse, `.` components and a
/// trailing slash drop, exactly two leading slashes stay, and an empty path is `.`.
pub fn path_str(s: &str) -> String {
    let root = if s.starts_with("//") && !s.starts_with("///") {
        "//"
    } else if s.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = s
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    let joined = format!("{root}{}", parts.join("/"));
    if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

/// Whether Python's `str.isprintable()` is true for `ch` (CPython 3.14.8 tables).
fn is_printable(ch: char) -> bool {
    let code = u32::from(ch);
    let idx = printable::NONPRINTABLE.partition_point(|&(_, hi)| hi < code);
    !printable::NONPRINTABLE
        .get(idx)
        .is_some_and(|&(lo, _)| lo <= code)
}

/// Python's `repr(s)` for a `str` (CPython `unicode_repr`).
///
/// Single quotes unless `s` holds a single quote and no double quote. Backslashes and the
/// chosen quote are escaped; `\t`, `\n`, `\r` use their short escapes; other
/// non-printable characters use `\xhh`, `\uhhhh`, or `\Uhhhhhhhh`.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        let code = u32::from(ch);
        if ch == quote || ch == '\\' {
            out.push('\\');
            out.push(ch);
        } else if ch == '\t' {
            out.push_str("\\t");
        } else if ch == '\n' {
            out.push_str("\\n");
        } else if ch == '\r' {
            out.push_str("\\r");
        } else if code < 0x20 || code == 0x7f {
            out.push_str(&format!("\\x{code:02x}"));
        } else if code < 0x7f || is_printable(ch) {
            out.push(ch);
        } else if code <= 0xff {
            out.push_str(&format!("\\x{code:02x}"));
        } else if code <= 0xffff {
            out.push_str(&format!("\\u{code:04x}"));
        } else {
            out.push_str(&format!("\\U{code:08x}"));
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_str_matches_python() {
        assert_eq!(repr_str("1-indexed"), "'1-indexed'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(repr_str("tab\there\\"), "'tab\\there\\\\'");
        assert_eq!(
            repr_str("\u{1}\u{7f}\u{a0}é\u{ad}"),
            "'\\x01\\x7f\\xa0é\\xad'"
        );
        assert_eq!(repr_str("\u{2028}\u{378}"), "'\\u2028\\u0378'");
        assert_eq!(repr_str("\u{e0001}"), "'\\U000e0001'");
        assert_eq!(repr_str("日本"), "'日本'");
    }
}
