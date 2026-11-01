//! The Python text semantics the readers copy, against CPython 3.14.8's own verdicts
//! (`tests/integration/data/pytext/`, from `scripts/testdata/dump_pytext.py`).

use std::fs;
use std::path::{Path, PathBuf};

use crate::harness::{as_array, as_str, as_u64, read_json, testdata_dir};
use moteval::benchmarks::motchallenge::read_seq_length;
use moteval::formats::pytext::{
    PyInt, py_float, py_int, py_name, py_stem, py_str, split_whitespace, splitlines, strip,
    utf8_error_message,
};
use serde_json::Value as Json;

fn corpus(topic: &str) -> Vec<Json> {
    let json = read_json(&testdata_dir().join("pytext").join(format!("{topic}.json")));
    as_array(&json).clone()
}

fn strings(json: &Json) -> Vec<&str> {
    as_array(json).iter().map(as_str).collect()
}

/// A fresh, empty scratch directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("moteval-{name}-{}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("remove old scratch dir");
    }
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[test]
fn float_and_int_accept_and_reject_what_python_does() {
    let cases = corpus("numbers");
    assert!(cases.len() > 2200);
    for case in &cases {
        let input = as_str(&case["input"]);
        let got = py_float(input);
        match case["float"].get("ok") {
            Some(bits) => {
                let got = got.unwrap_or_else(|| panic!("float({input:?}): Python accepts it"));
                assert_eq!(got.to_bits(), as_u64(bits), "float({input:?})");
            }
            None => assert!(
                got.is_none(),
                "float({input:?}): Python rejects it, got {got:?}"
            ),
        }
        let got = py_int(input);
        let expected = &case["int"];
        if let Some(value) = expected.get("ok") {
            let value = value.as_i64().expect("an i64");
            assert_eq!(got, Ok(PyInt::Value(value)), "int({input:?})");
        } else if let Some(sign) = expected.get("out_of_range") {
            let negative = as_str(sign) == "negative";
            assert_eq!(got, Ok(PyInt::OutOfRange { negative }), "int({input:?})");
        } else {
            let err = got.expect_err(&format!("int({input:?}): Python rejects it"));
            assert_eq!(
                moteval::formats::pytext::int_error_message(input, err),
                as_str(&expected["message"]),
                "int({input:?}) message"
            );
        }
    }
}

#[test]
fn strip_split_and_splitlines_match_python() {
    for case in &corpus("text") {
        let input = as_str(&case["input"]);
        assert_eq!(strip(input), as_str(&case["strip"]), "{input:?}.strip()");
        assert_eq!(
            split_whitespace(input),
            strings(&case["split"]),
            "{input:?}.split()"
        );
        assert_eq!(
            splitlines(input),
            strings(&case["splitlines"]),
            "{input:?}.splitlines()"
        );
    }
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

#[test]
fn utf8_errors_match_python() {
    for case in &corpus("utf8") {
        let data = hex_bytes(as_str(&case["hex"]));
        let got = utf8_error_message(&data);
        match case["decode"].get("message") {
            Some(message) => assert_eq!(got.as_deref(), Some(as_str(message)), "{data:x?}"),
            None => assert_eq!(got, None, "{data:x?}"),
        }
    }
}

#[test]
fn path_str_name_and_stem_match_pathlib() {
    for case in &corpus("paths") {
        let input = as_str(&case["input"]);
        let path = Path::new(input);
        assert_eq!(py_str(path), as_str(&case["str"]), "str(Path({input:?}))");
        let name = py_name(path);
        assert_eq!(name, as_str(&case["name"]), "Path({input:?}).name");
        assert_eq!(
            py_stem(&name),
            as_str(&case["stem"]),
            "Path({input:?}).stem"
        );
    }
}

#[test]
fn read_seq_length_matches_configparser() {
    let cases = corpus("ini");
    assert!(cases.len() > 1500);
    let root = scratch("ini");
    for (k, case) in cases.iter().enumerate() {
        let text = as_str(&case["text"]);
        let seq_dir = root.join(format!("seq-{k}"));
        fs::create_dir_all(&seq_dir).expect("create seq dir");
        fs::write(seq_dir.join("seqinfo.ini"), text).expect("write seqinfo.ini");
        let got = read_seq_length(&seq_dir);
        let result = &case["result"];
        if let Some(value) = result.get("ok") {
            let value = usize::try_from(as_u64(value)).expect("usize");
            assert_eq!(got.as_ref().ok(), Some(&value), "{text:?}: {got:?}");
        } else if result.get("unsupported").is_some() {
            let err = got.expect_err("Rust cannot hold this length");
            assert!(
                matches!(err, moteval::benchmarks::BenchmarkError::Unsupported(_)),
                "{text:?}: {err:?}"
            );
        } else {
            let err = got.expect_err(&format!("{text:?}: Python raises"));
            let message = err.to_string().replace(&py_str(&seq_dir), "<SEQ>");
            assert_eq!(message, as_str(&result["message"]), "{text:?}");
        }
    }
    fs::remove_dir_all(&root).expect("remove scratch dir");
}
