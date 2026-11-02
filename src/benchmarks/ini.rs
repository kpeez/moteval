use crate::formats::pytext::{is_space, strip};
use crate::pyfmt::repr_str;

const DEFAULT_SECTION: &str = "DEFAULT";
const MAX_INTERPOLATION_DEPTH: usize = 10;

#[derive(Debug, Default)]
pub struct ConfigParser {
    defaults: Vec<(String, String)>,
    sections: Vec<(String, Vec<(String, String)>)>,
}

#[derive(Clone, Copy, PartialEq)]
enum Cursor {
    Defaults,
    Section(usize),
}

type RawOptions = Vec<(String, Vec<String>)>;

fn find<'a, T>(items: &'a [(String, T)], key: &str) -> Option<&'a T> {
    items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn optionxform(name: &str) -> String {
    name.to_lowercase()
}

fn line_tag(lineno: usize) -> String {
    format!(" [line {lineno:2}]")
}

pub fn posix_path_repr(path: &str) -> String {
    format!("PosixPath({})", repr_str(path))
}

fn file_lines(text: &str) -> Vec<String> {
    let translated = text.replace("\r\n", "\n").replace('\r', "\n");
    translated
        .split_inclusive('\n')
        .map(str::to_owned)
        .collect()
}

fn clean(trimmed: &str) -> &str {
    if trimmed.starts_with('#') || trimmed.starts_with(';') {
        ""
    } else {
        trimmed
    }
}

impl ConfigParser {
    pub fn read(text: &str, source: &str) -> Result<Self, String> {
        let mut defaults: RawOptions = Vec::new();
        let mut sections: Vec<(String, RawOptions)> = Vec::new();
        let mut added_sections: Vec<String> = Vec::new();
        let mut added_options: Vec<(String, String)> = Vec::new();
        let mut cursect: Option<Cursor> = None;
        let mut sectname: Option<String> = None;
        let mut optname: Option<String> = None;
        let mut indent_level: usize = 0;
        let mut errors: Vec<(usize, String)> = Vec::new();

        for (index, line) in file_lines(text).iter().enumerate() {
            let lineno = index + 1;
            let trimmed = strip(line);
            let clean_line = clean(trimmed);
            let has_comments = trimmed != clean_line;
            if clean_line.is_empty() {
                // CPython configparser `empty_lines_in_values`: a blank line joins the
                // current value; a comment-only line does not.
                if !has_comments
                    && let (Some(cursor), Some(name)) = (cursect, optname.as_deref())
                    && !name.is_empty()
                {
                    value_mut(cursor, name, &mut defaults, &mut sections).push(String::new());
                }
                continue;
            }
            let cur_indent_level = line.chars().position(|c| !is_space(c)).unwrap_or(0);

            if let (Some(cursor), Some(name)) = (cursect, optname.as_deref())
                && !name.is_empty()
                && cur_indent_level > indent_level
            {
                value_mut(cursor, name, &mut defaults, &mut sections).push(clean_line.to_owned());
                continue;
            }

            indent_level = cur_indent_level;
            let header = section_header(clean_line);
            if header.is_none() && cursect.is_none() {
                return Err(format!(
                    "File contains no section headers.\nfile: {source}, line: {lineno}\n{}",
                    repr_str(line)
                ));
            }
            if let Some(name) = header {
                if let Some(pos) = sections.iter().position(|(s, _)| s == name) {
                    if added_sections.iter().any(|s| s == name) {
                        return Err(format!(
                            "While reading from {source}{}: section {} already exists",
                            line_tag(lineno),
                            repr_str(name)
                        ));
                    }
                    cursect = Some(Cursor::Section(pos));
                    added_sections.push(name.to_owned());
                } else if name == DEFAULT_SECTION {
                    cursect = Some(Cursor::Defaults);
                } else {
                    sections.push((name.to_owned(), Vec::new()));
                    cursect = Some(Cursor::Section(sections.len() - 1));
                    added_sections.push(name.to_owned());
                }
                sectname = Some(name.to_owned());
                optname = None;
                continue;
            }

            let Some((option, value)) = split_option(clean_line) else {
                errors.push((lineno, line.clone()));
                continue;
            };
            if option.is_empty() {
                errors.push((lineno, line.clone()));
            }
            let name = optionxform(option);
            let section = sectname
                .clone()
                .expect("an option follows a section header");
            if added_options
                .iter()
                .any(|(s, o)| *s == section && *o == name)
            {
                return Err(format!(
                    "While reading from {source}{}: option {} in section {} already exists",
                    line_tag(lineno),
                    repr_str(&name),
                    repr_str(&section)
                ));
            }
            added_options.push((section, name.clone()));
            let cursor = cursect.expect("an option follows a section header");
            let options = match cursor {
                Cursor::Defaults => &mut defaults,
                Cursor::Section(pos) => &mut sections[pos].1,
            };
            let entry = vec![strip(value).to_owned()];
            match options.iter_mut().find(|(k, _)| *k == name) {
                Some(slot) => slot.1 = entry,
                None => options.push((name.clone(), entry)),
            }
            optname = Some(name);
        }

        if let Some(((first_lineno, first_line), rest)) = errors.split_first() {
            let mut message = format!(
                "Source contains parsing errors: {source}\n\t[line {first_lineno:2}]: {}",
                repr_str(first_line)
            );
            for (lineno, line) in rest {
                message.push_str(&format!("\n\t[line {lineno:2}]: {}", repr_str(line)));
            }
            return Err(message);
        }

        let join = |options: RawOptions| -> Vec<(String, String)> {
            options
                .into_iter()
                .map(|(k, lines)| (k, lines.join("\n").trim_end_matches(is_space).to_owned()))
                .collect()
        };
        Ok(Self {
            defaults: join(defaults),
            sections: sections
                .into_iter()
                .map(|(name, options)| (name, join(options)))
                .collect(),
        })
    }

    pub fn has_option(&self, section: &str, option: &str) -> bool {
        let option = optionxform(option);
        match find(&self.sections, section) {
            None => false,
            Some(options) => {
                find(options, &option).is_some() || find(&self.defaults, &option).is_some()
            }
        }
    }

    pub fn get(&self, section: &str, option: &str) -> Result<String, String> {
        let option = optionxform(option);
        let options = find(&self.sections, section).expect("has_option checked the section");
        let lookup = |name: &str| find(options, name).or_else(|| find(&self.defaults, name));
        let raw = lookup(&option).expect("has_option checked the option");
        let mut out = String::new();
        interpolate(&option, section, raw, raw, &lookup, &mut out, 1)?;
        Ok(out)
    }
}

/// CPython configparser's greedy `SECTCRE`: the header runs to the last `]`.
fn section_header(clean: &str) -> Option<&str> {
    let rest = clean.strip_prefix('[')?;
    let close = rest.rfind(']')?;
    (close > 0).then(|| &rest[..close])
}

fn split_option(clean: &str) -> Option<(&str, &str)> {
    let at = clean.find(['=', ':'])?;
    let option = clean[..at].trim_end_matches(is_space);
    let value = clean[at + 1..].trim_start_matches(is_space);
    Some((option, value))
}

fn value_mut<'a>(
    cursor: Cursor,
    name: &str,
    defaults: &'a mut RawOptions,
    sections: &'a mut [(String, RawOptions)],
) -> &'a mut Vec<String> {
    let options = match cursor {
        Cursor::Defaults => defaults,
        Cursor::Section(pos) => &mut sections[pos].1,
    };
    &mut options
        .iter_mut()
        .find(|(k, _)| k == name)
        .expect("the current option exists")
        .1
}

fn interpolate<'a>(
    option: &str,
    section: &str,
    rawval: &str,
    mut rest: &'a str,
    lookup: &dyn Fn(&str) -> Option<&'a String>,
    out: &mut String,
    depth: usize,
) -> Result<(), String> {
    if depth > MAX_INTERPOLATION_DEPTH {
        return Err(format!(
            "Recursion limit exceeded in value substitution: option {} in section {} \
             contains an interpolation key which cannot be substituted in \
             {MAX_INTERPOLATION_DEPTH} steps. Raw value: {}",
            repr_str(option),
            repr_str(section),
            repr_str(rawval)
        ));
    }
    while !rest.is_empty() {
        let Some(p) = rest.find('%') else {
            out.push_str(rest);
            return Ok(());
        };
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let next = rest[1..].chars().next();
        if next == Some('%') {
            out.push('%');
            rest = &rest[2..];
        } else if next == Some('(') {
            let close = rest[2..].find(')').map(|i| i + 2);
            let name = match close {
                Some(c) if c > 2 && rest[c + 1..].starts_with('s') => &rest[2..c],
                _ => {
                    return Err(format!(
                        "bad interpolation variable reference {}",
                        repr_str(rest)
                    ));
                }
            };
            let var = optionxform(name);
            rest = &rest[close.expect("matched") + 2..];
            let Some(value) = lookup(&var) else {
                return Err(format!(
                    "Bad value substitution: option {} in section {} contains an \
                     interpolation key {} which is not a valid option name. Raw value: {}",
                    repr_str(option),
                    repr_str(section),
                    repr_str(&var),
                    repr_str(rawval)
                ));
            };
            if value.contains('%') {
                interpolate(option, section, rawval, value, lookup, out, depth + 1)?;
            } else {
                out.push_str(value);
            }
        } else {
            return Err(format!(
                "'%' must be followed by '%' or '(', found: {}",
                repr_str(rest)
            ));
        }
    }
    Ok(())
}

/// CPython's `TextIOWrapper` decodes in 8192-byte chunks, so error positions are
/// chunk-relative.
pub fn decode_chunked(bytes: &[u8]) -> Result<String, String> {
    const CHUNK: usize = 8192;
    let mut text = String::new();
    let mut pending: Vec<u8> = Vec::new();
    for chunk in bytes.chunks(CHUNK) {
        let mut data = std::mem::take(&mut pending);
        data.extend_from_slice(chunk);
        match std::str::from_utf8(&data) {
            Ok(s) => text.push_str(s),
            Err(err) if err.error_len().is_none() => {
                let valid = err.valid_up_to();
                text.push_str(std::str::from_utf8(&data[..valid]).expect("valid prefix"));
                pending = data[valid..].to_vec();
            }
            Err(_) => {
                return Err(crate::formats::pytext::utf8_error_message(&data)
                    .expect("invalid UTF-8 has an error"));
            }
        }
    }
    if !pending.is_empty() {
        return Err(crate::formats::pytext::utf8_error_message(&pending)
            .expect("a truncated sequence is invalid"));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::pytext_corpus;

    #[test]
    fn chunked_decode_errors_match_text_io_wrapper() {
        let cases = pytext_corpus("utf8_chunked");
        assert!(cases.len() > 100);
        for case in &cases {
            let size = case["size"].as_u64().expect("size") as usize;
            let at = case["at"].as_u64().expect("at") as usize;
            let hex = case["bad"].as_str().expect("bad");
            let bad: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
                .collect();
            let mut body = vec![b'x'; size];
            let end = (at + bad.len()).min(size);
            body[at..end].copy_from_slice(&bad[..end - at]);
            let got = decode_chunked(&body);
            match case["iterate"].get("message") {
                Some(message) => {
                    assert_eq!(got.err().as_deref(), message.as_str(), "{size} {hex} @{at}")
                }
                None => assert!(got.is_ok(), "{size} {hex} @{at}: {got:?}"),
            }
        }
    }
}
