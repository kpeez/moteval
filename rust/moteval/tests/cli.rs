//! The `moteval` binary against the Python CLI (`rust/scripts/dump_cli.py`): for every
//! dumped argument list, the same stdout, stderr, and exit code, byte for byte.
//!
//! Every case stops before a dataset is loaded: argparse parsing and errors, `--metrics`
//! resolution, the `--gt` and `--pred` checks, and the `tests/test_cli.py` cases that need
//! no dataset and no metric. Cases that load and score data need the loader and metric
//! packages; they move to the integration gate.

mod harness;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use harness::{as_array, as_i64, as_str, read_json, testdata_dir};

/// A fresh copy of the dump's fixture tree, as the working directory of each case.
fn fixture(spec: &serde_json::Value) -> PathBuf {
    let root = std::env::temp_dir().join(format!("moteval-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    for dir in as_array(&spec["dirs"]) {
        fs::create_dir_all(root.join(as_str(dir))).unwrap();
    }
    for file in as_array(&spec["files"]) {
        fs::write(root.join(as_str(file)), "").unwrap();
    }
    root
}

#[test]
fn binary_matches_python_cli_byte_for_byte() {
    let data = read_json(&testdata_dir().join("cli.json"));
    let cwd = fixture(&data["fixture"]);
    let cases = as_array(&data["cases"]);
    for case in cases {
        let argv: Vec<&str> = as_array(&case["argv"]).iter().map(as_str).collect();
        let output = Command::new(env!("CARGO_BIN_EXE_moteval"))
            .args(&argv)
            .current_dir(&cwd)
            .env("COLUMNS", "80")
            .output()
            .expect("the moteval binary runs");
        let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
        let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
        assert_eq!(stderr, as_str(&case["stderr"]), "stderr of {argv:?}");
        assert_eq!(stdout, as_str(&case["stdout"]), "stdout of {argv:?}");
        assert_eq!(
            i64::from(output.status.code().expect("an exit code")),
            as_i64(&case["exit"]),
            "exit code of {argv:?}"
        );
    }
    fs::remove_dir_all(&cwd).unwrap();
    assert!(cases.len() >= 100, "only {} cases", cases.len());
    println!("cli: {} argument lists match the Python CLI", cases.len());
}
