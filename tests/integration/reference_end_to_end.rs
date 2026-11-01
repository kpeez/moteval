//! The integration gate: the whole Rust pipeline on all nine local benchmarks equals
//! `tests/fixtures/reference_benchmarks.json`, bit for bit.
//!
//! For each `<benchmark>_<split>` case, Rust loads the raw ground truth with
//! `benchmarks::load_dataset` (from `MOTEVAL_DATA_ROOT`), reads the seeded predictions
//! `dump_loaders.py` wrote (`loaders_real/<case>/pred/`, the files
//! `scripts/reference_results.py` scores), and runs `eval::evaluate` with the metrics the
//! reference scored. Nothing Python computed enters the run.
//!
//! - Combined scores equal the reference fixture. `dump_scenarios.py --real-data` stores
//!   the fixture's numbers as float bits in `reference_benchmarks/<case>/case.json`
//!   (`oracle.combined`), after checking them against the fixture, because serde_json
//!   cannot read the fixture's `NaN` tokens.
//! - Per-sequence scores equal Python moteval's (`moteval.views[0].per_sequence` in the
//!   same file), in the same sequence and metric order.

use std::path::Path;

use crate::harness::{
    as_array, as_str, assert_field_order, assert_scores_bit_equal, decode_metric_scores,
    metric_by_name, read_json, real_case_names, real_testdata_dir,
};
use moteval::benchmarks;
use moteval::eval;

fn data_root() -> Option<std::path::PathBuf> {
    match std::env::var_os("MOTEVAL_DATA_ROOT") {
        Some(root) if !root.is_empty() && Path::new(&root).is_dir() => Some(root.into()),
        _ => {
            eprintln!(
                "SKIPPING REAL-DATA GATE: reference_end_to_end: MOTEVAL_DATA_ROOT is not set \
                 or not a directory"
            );
            None
        }
    }
}

/// The nine benchmarks and every split the reference fixture scores.
const EXPECTED_CASES: usize = 20;

#[test]
fn rust_pipeline_equals_the_reference_on_nine_benchmarks() {
    let (Some(testdata), Some(data_root)) =
        (real_testdata_dir("reference_end_to_end"), data_root())
    else {
        return;
    };
    let names = real_case_names(&testdata, "reference_benchmarks");
    assert_eq!(
        names.len(),
        EXPECTED_CASES,
        "reference_benchmarks cases under {}",
        testdata.display()
    );
    let mut fields = 0;
    let mut sequences = 0;
    let mut benchmarks_seen = std::collections::BTreeSet::new();
    for name in &names {
        let (benchmark, split) = name.split_once('_').expect("<benchmark>_<split>");
        let case = read_json(
            &testdata
                .join("reference_benchmarks")
                .join(name)
                .join("case.json"),
        );
        let pred_dir = testdata.join("loaders_real").join(name).join("pred");
        assert!(
            pred_dir.is_dir(),
            "{name}: {} missing - run scripts/testdata/dump_loaders.py",
            pred_dir.display()
        );
        let dataset =
            benchmarks::load_dataset(benchmark, Some(&data_root.join(benchmark)), Some(split))
                .unwrap_or_else(|err| panic!("{name}: load_dataset: {err}"));
        let metrics: Vec<_> = as_array(&case["metrics"])
            .iter()
            .map(|m| metric_by_name(as_str(m)))
            .collect();
        let result = eval::evaluate(&dataset, &pred_dir, &metrics)
            .unwrap_or_else(|err| panic!("{name}: evaluate: {err}"));
        assert!(
            !result.is_multi_class(),
            "{name}: the reference is single-class"
        );

        for (metric, want) in decode_metric_scores(&case["oracle"]["combined"]) {
            let got = result
                .combined
                .get(&metric)
                .unwrap_or_else(|| panic!("{name}: no combined {metric}"));
            assert_scores_bit_equal(&want, got, &format!("{name} combined {metric}"));
            fields += want.len();
        }

        let python = as_array(&case["moteval"]["views"][0]["per_sequence"]);
        let keys: Vec<&str> = result
            .per_sequence
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        let want_keys: Vec<&str> = python.iter().map(|pair| as_str(&pair[0])).collect();
        assert_eq!(keys, want_keys, "{name}: sequence order");
        for (pair, (seq, got)) in python.iter().zip(&result.per_sequence) {
            // The dump keeps a metric with no declared per-sequence field (TrackMAP) as an
            // empty map; `evaluate` gives it no per-sequence entry, as Python's does.
            let want: Vec<_> = decode_metric_scores(&pair[1])
                .into_iter()
                .filter(|(_, scores)| !scores.is_empty())
                .collect();
            let got_metrics: Vec<&str> = got.iter().map(|(m, _)| m).collect();
            let want_metrics: Vec<&str> = want.iter().map(|(m, _)| m.as_str()).collect();
            assert_eq!(got_metrics, want_metrics, "{name}/{seq}: metric order");
            for (metric, want) in &want {
                let got = got.get(metric).expect("metric present");
                let label = format!("{name}/{seq} {metric}");
                assert_field_order(want, got, &label);
                assert_scores_bit_equal(want, got, &label);
            }
            sequences += 1;
        }
        benchmarks_seen.insert(benchmark.to_string());
        eprintln!("{name}: {} sequences equal", python.len());
    }
    assert_eq!(benchmarks_seen.len(), 9, "benchmarks: {benchmarks_seen:?}");
    eprintln!(
        "reference_end_to_end: {} cases on {} benchmarks, {fields} combined fields equal the \
         reference, {sequences} sequences equal Python",
        names.len(),
        benchmarks_seen.len()
    );
}
