//! Python-compatible text output.
//!
//! The results-cli-bindings package adds `repr(float)` (shortest round-trip digits,
//! exponent form below `1e-4` and from `1e16`, written `1e-05`), `format(x, "1.5g")`,
//! `json.dumps` defaults (`", "` and `": "` separators, `NaN`, `Infinity`, `-Infinity`),
//! and `csv.writer` defaults (`\r\n` line ends, minimal quoting), each tested against
//! strings dumped from Python.
//!
//! `repr_str` is here first because error messages across the crate quote names with
//! Python's `{name!r}`.

mod printable;

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
