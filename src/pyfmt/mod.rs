mod printable;

fn digits_and_exponent(x: f64, precision: Option<usize>) -> (String, i32) {
    debug_assert!(x.is_finite() && x.is_sign_positive());
    match precision {
        Some(p) => split_sci(&format!("{x:.*e}", p.saturating_sub(1))),
        None => {
            // Rust's `{:e}` breaks a tie between two shortest round-trip strings upward;
            // CPython's dtoa breaks it to even (`repr(4092142886402.53125)` ends `.5312`).
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

pub fn repr_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let (digits, exponent) = digits_and_exponent(x.abs(), None);
    let decpt = exponent + 1;
    // CPython's float repr thresholds (`float_repr_style == 'short'`).
    layout(
        x.is_sign_negative(),
        &digits,
        decpt,
        decpt <= -4 || decpt > 16,
        true,
    )
}

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

pub fn csv_row<S: AsRef<str>>(fields: &[S]) -> String {
    let mut out = String::new();
    for (index, field) in fields.iter().enumerate() {
        let field = field.as_ref();
        if index > 0 {
            out.push(',');
        }
        // CPython's `csv.writer` quotes a lone empty field so the row reads back.
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

/// CPython `str(pathlib.PurePosixPath(s))`, which keeps exactly two leading slashes.
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

fn is_printable(ch: char) -> bool {
    let code = u32::from(ch);
    let idx = printable::NONPRINTABLE.partition_point(|&(_, hi)| hi < code);
    !printable::NONPRINTABLE
        .get(idx)
        .is_some_and(|&(lo, _)| lo <= code)
}

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
