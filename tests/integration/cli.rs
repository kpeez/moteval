use std::fs;
use std::path::PathBuf;
use std::process::Command;

use crate::harness::{as_array, as_i64, as_str, read_json, testdata_dir};

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
