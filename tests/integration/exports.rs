use std::fs;
use std::path::PathBuf;

use crate::harness::{as_array, as_i64, as_str, decode_metric_scores, read_json, testdata_dir};
use moteval::cli::{TableError, table};
use moteval::eval::evaluate;
use moteval::metrics::{Clear, Count, Hota, Identity, JAndF, Metric, TrackMap};
use moteval::model::{FrameConvention, GtSequence, MOTDataset, MaskGtSequence, Protocol, Sequence};
use moteval::results::{ClassResult, EvaluationResult, MetricScores, to_csv, to_json};
use serde_json::Value as Json;

fn data() -> Json {
    read_json(&testdata_dir().join("exports.json"))
}

fn metric_scores(json: &Json) -> MetricScores {
    let mut scores = MetricScores::new();
    for (metric, fields) in decode_metric_scores(json) {
        scores.insert(metric, fields);
    }
    scores
}

fn per_sequence(json: &Json) -> Vec<(String, MetricScores)> {
    as_array(json)
        .iter()
        .map(|pair| (as_str(&pair[0]).to_string(), metric_scores(&pair[1])))
        .collect()
}

fn decode_result(json: &Json) -> EvaluationResult {
    EvaluationResult {
        per_sequence: per_sequence(&json["per_sequence"]),
        combined: metric_scores(&json["combined"]),
        per_class: as_array(&json["per_class"])
            .iter()
            .map(|pair| {
                (
                    as_i64(&pair[0]),
                    ClassResult {
                        per_sequence: per_sequence(&pair[1]["per_sequence"]),
                        combined: metric_scores(&pair[1]["combined"]),
                    },
                )
            })
            .collect(),
        class_averaged: metric_scores(&json["class_averaged"]),
        det_averaged: metric_scores(&json["det_averaged"]),
    }
}

fn assert_same_text(want: &str, got: &str, label: &str) {
    if want == got {
        return;
    }
    let line = want
        .split('\n')
        .zip(got.split('\n'))
        .position(|(w, g)| w != g)
        .unwrap_or_else(|| want.split('\n').count().min(got.split('\n').count()));
    panic!(
        "{label}: differs from Python at line {line}\n python: {:?}\n   rust: {:?}",
        want.split('\n').nth(line),
        got.split('\n').nth(line)
    );
}

#[test]
fn table_json_and_csv_equal_python_bytes() {
    let data = data();
    let cases = as_array(&data["cases"]);
    for case in cases {
        let name = as_str(&case["name"]);
        let result = decode_result(&case["result"]);
        let json = to_json(&result, as_str(&case["dataset"]), as_str(&case["split"]));
        assert_same_text(as_str(&case["json"]), &json, &format!("{name} json"));
        assert_same_text(
            as_str(&case["csv"]),
            &to_csv(&result),
            &format!("{name} csv"),
        );
        let want = &case["table"];
        match table(&result) {
            Ok(text) => {
                assert!(
                    want.get("text").is_some(),
                    "{name}: Python's table raised {want}, Rust's gave a table"
                );
                assert_same_text(as_str(&want["text"]), &text, &format!("{name} table"));
            }
            Err(err) => {
                let kind = match err {
                    TableError::NanToInteger => "ValueError",
                    TableError::InfinityToInteger => "OverflowError",
                    TableError::MissingScore { .. } => "KeyError",
                };
                assert_eq!(
                    want.get("error").map(as_str),
                    Some(kind),
                    "{name}: table error {err}"
                );
                if kind != "KeyError" {
                    assert_eq!(as_str(&want["message"]), err.to_string(), "{name}");
                }
            }
        }
    }
    assert!(cases.len() >= 12, "only {} cases", cases.len());
}

fn metric(name: &str) -> Box<dyn Metric> {
    match name {
        "HOTA" => Box::new(Hota),
        "CLEAR" => Box::new(Clear),
        "Identity" => Box::new(Identity),
        "Count" => Box::new(Count),
        "JAndF" => Box::new(JAndF),
        "TrackMAP" => Box::new(TrackMap),
        other => panic!("unknown metric {other}"),
    }
}

fn dataset(spec: &Json) -> MOTDataset {
    let sequences = as_array(&spec["sequences"])
        .iter()
        .map(|seq| {
            let name = as_str(&seq["name"]).to_string();
            match as_str(&seq["kind"]) {
                "masks" => Sequence::Masks(MaskGtSequence {
                    name,
                    num_timesteps: 1,
                    tracks: Vec::new(),
                    ignore_regions: Vec::new(),
                }),
                _ => Sequence::Boxes(GtSequence {
                    name,
                    num_timesteps: 1,
                    tracks: Vec::new(),
                    ignore_regions: Vec::new(),
                }),
            }
        })
        .collect();
    let classes = as_array(&spec["eval_classes"]).iter().map(as_i64).collect();
    MOTDataset {
        name: as_str(&spec["name"]).to_string(),
        split: "val".to_string(),
        sequences,
        protocol: Protocol::new(
            as_str(&spec["protocol"]),
            FrameConvention::new("1-indexed", 1),
            classes,
        ),
    }
}

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("moteval-exports-{}-{label}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn errors_before_scoring_match_python_messages() {
    let data = data();
    let cases = as_array(&data["errors"]);
    for case in cases {
        let name = as_str(&case["name"]);
        let root = temp_dir(name);
        let pred_dir = root.join("preds");
        fs::create_dir_all(&pred_dir).unwrap();
        for stem in as_array(&case["files"]) {
            fs::write(pred_dir.join(format!("{}.txt", as_str(stem))), "").unwrap();
        }
        let metrics: Vec<Box<dyn Metric>> = as_array(&case["metrics"])
            .iter()
            .map(|m| metric(as_str(m)))
            .collect();
        let given = PathBuf::from(format!("{}//./", pred_dir.display()));
        let err = match evaluate(&dataset(&case["dataset"]), &given, &metrics) {
            Ok(_) => panic!("{name}: evaluate did not fail"),
            Err(err) => err,
        };
        let message = err
            .to_string()
            .replace(&pred_dir.display().to_string(), "<PRED>");
        assert_eq!(message, as_str(&case["message"]), "{name}");
        fs::remove_dir_all(&root).unwrap();
    }
    assert!(cases.len() >= 10, "only {} error cases", cases.len());
}
