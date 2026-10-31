//! CLEAR, Identity, and Count against the TrackEval oracle and Python moteval, bit for
//! bit.
//!
//! Every case runs as `evaluate` runs it: each sequence scored, each class view combined,
//! and the class combiners over the views. Each result is compared with the oracle
//! fields the fixture freezes (`check_case_against_oracle`) and with Python moteval's own
//! dumped scores: per sequence, per view combined, class-averaged, and det-averaged,
//! every field and its dict order.
//!
//! MOTAL takes `np.log10(IDSW)` through `numeric::log10`, which makes numpy's choice
//! between SVML and libm. Some cases here have an `IDSW` (11, 53, 119, 174) where the two
//! differ, but their `TP - FP` is large enough that MOTAL is the same either way.
//! `metrics::clear::tests::motal_takes_numpy_log10` covers counts where it is not.

mod harness;

use std::path::Path;

use harness::{
    Case, check_case_against_oracle, load_case, load_real_case, load_synthetic_case,
    real_case_names, real_testdata_dir, same_bits, synthetic_case_names, testdata_dir,
};
use moteval::metrics::{Clear, Count, Identity, Metric, Scores};

const METRICS: [&dyn Metric; 3] = [&Clear, &Identity, &Count];

/// Every difference between `expected` and `got`: a field missing, a shape or bit
/// difference, and (unless `got` is reduced to the expected fields) the field order.
fn differences(expected: &Scores, got: &Scores, label: &str) -> Vec<String> {
    let mut out = Vec::new();
    let want_order: Vec<&str> = expected.iter().map(|(f, _)| f).collect();
    let have_order: Vec<&str> = got.iter().map(|(f, _)| f).collect();
    if want_order != have_order {
        out.push(format!(
            "{label}: field order {have_order:?} != Python {want_order:?}"
        ));
    }
    for (field, want) in expected.iter() {
        let Some(have) = got.get(field) else {
            out.push(format!("{label}.{field}: missing"));
            continue;
        };
        if have.shape() != want.shape() {
            out.push(format!("{label}.{field}: shape differs"));
            continue;
        }
        for (i, (&w, &h)) in want.as_slice().iter().zip(have.as_slice()).enumerate() {
            if !same_bits(w, h) {
                out.push(format!(
                    "{label}.{field}[{i}]: got {h:e} ({:#018x}), Python {w:e} ({:#018x})",
                    h.to_bits(),
                    w.to_bits()
                ));
            }
        }
    }
    out
}

fn find<'a>(by_metric: &'a [(String, Scores)], metric: &str) -> &'a Scores {
    &by_metric
        .iter()
        .find(|(name, _)| name == metric)
        .unwrap_or_else(|| panic!("no Python scores for {metric}"))
        .1
}

/// Run `metric` over `case` as `evaluate` does and list every difference from Python
/// moteval's dumped scores. Returns the differences and the number of score maps
/// compared.
fn compare_with_python(case: &Case, metric: &dyn Metric) -> (Vec<String>, usize) {
    let name = metric.name();
    let label = format!("{}/{}.{name}", case.fixture, case.case);
    let mut diffs = Vec::new();
    let mut compared = 0;
    let mut combined = Vec::new();
    for (v, view) in case.views.iter().enumerate() {
        let mut per_sequence = Vec::new();
        for (seq, (key, python)) in view.sequences.iter().zip(&view.moteval_per_sequence) {
            assert_eq!(&seq.key, key, "{label}: sequence order");
            let got = metric
                .eval_sequence(&seq.data)
                .unwrap_or_else(|err| panic!("{label}: {key}: {err}"));
            let want = find(python, name);
            // Real-data dumps hold the declared fields only.
            let got_cmp = if want.len() == got.len() {
                got.clone()
            } else {
                got.declared(metric.fields())
            };
            diffs.extend(differences(
                want,
                &got_cmp,
                &format!("{label} view{v} {key}"),
            ));
            compared += 1;
            per_sequence.push(got);
        }
        let refs: Vec<&Scores> = per_sequence.iter().collect();
        let got = metric
            .combine_sequences(&refs)
            .unwrap_or_else(|err| panic!("{label}: combine_sequences: {err}"));
        diffs.extend(differences(
            find(&view.moteval_combined, name),
            &got,
            &format!("{label} view{v} combined"),
        ));
        compared += 1;
        combined.push(got);
    }
    if case.procedure == "per_class" && case.views.len() > 1 {
        let refs: Vec<&Scores> = combined.iter().collect();
        let class_averaged = metric
            .combine_classes_class_averaged(&refs)
            .unwrap_or_else(|err| panic!("{label}: class_averaged: {err}"));
        let det_averaged = metric
            .combine_classes_det_averaged(&refs)
            .unwrap_or_else(|err| panic!("{label}: det_averaged: {err}"));
        diffs.extend(differences(
            find(&case.moteval_class_averaged, name),
            &class_averaged,
            &format!("{label} class_averaged"),
        ));
        diffs.extend(differences(
            find(&case.moteval_det_averaged, name),
            &det_averaged,
            &format!("{label} det_averaged"),
        ));
        compared += 2;
    }
    (diffs, compared)
}

/// Check every CLEAR, Identity, and Count entry of `case` against the oracle and Python.
/// Returns (oracle fields checked, score maps compared with Python).
fn check_gate_case(case: &Case) -> (usize, usize) {
    let mut oracle_fields = 0;
    let mut compared = 0;
    for metric in METRICS {
        if !case.metrics.iter().any(|m| m == metric.name()) {
            continue;
        }
        let checked = check_case_against_oracle(case, metric);
        assert!(
            checked > 0,
            "{}/{}: no oracle fields for {}",
            case.fixture,
            case.case,
            metric.name()
        );
        oracle_fields += checked;
        let (diffs, n) = compare_with_python(case, metric);
        assert!(diffs.is_empty(), "{}", diffs.join("\n"));
        compared += n;
    }
    (oracle_fields, compared)
}

#[test]
fn synthetic_box_and_mots_match_oracle_and_python() {
    let mut cases = 0;
    let mut oracle_fields = 0;
    let mut compared = 0;
    for fixture in ["synthetic_box", "synthetic_mots"] {
        for name in synthetic_case_names(fixture) {
            let case = load_synthetic_case(fixture, &name);
            if !case.metrics.iter().any(|m| m == "CLEAR") {
                continue;
            }
            let (fields, n) = check_gate_case(&case);
            oracle_fields += fields;
            compared += n;
            cases += 1;
        }
    }
    // synthetic_box: every case but the HOTA-only class-combiner case; synthetic_mots:
    // all seven.
    assert_eq!(cases, 17 + 7, "CLEAR/Identity/Count synthetic case count");
    eprintln!(
        "synthetic: {cases} cases, {oracle_fields} oracle fields, {compared} score maps \
         compared with Python"
    );
}

#[test]
fn real_data_matches_oracle_and_python() {
    let Some(root) = real_testdata_dir("CLEAR/Identity/Count real_data.json") else {
        return;
    };
    let mut cases = 0;
    for name in ["dancetrack_val", "sportsmot_val", "mots20"] {
        let Some(case) = load_real_case(&root, "real_data", name) else {
            continue;
        };
        let (fields, compared) = check_gate_case(&case);
        eprintln!(
            "real_data/{name}: {fields} oracle fields, {compared} score maps compared with \
             Python"
        );
        cases += 1;
    }
    eprintln!("real_data: {cases} of 3 cases checked");
}

fn fuzz_case_paths() -> Vec<std::path::PathBuf> {
    let dir = testdata_dir().join("fuzz").join("clear_identity_count");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("cannot list {}: {err}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    paths
}

/// Seeded random sequences (`rust/scripts/dump_clear_identity_count.py`): repeated ids
/// within a frame, similarities on each side of both thresholds, empty frames, and one
/// to four classes. Six more cases combine 8 to 12 classes of frames with 8 or more
/// matches, where numpy's pairwise mean and sum differ from a plain sum.
#[test]
fn seeded_cases_match_python() {
    let paths = fuzz_case_paths();
    assert_eq!(paths.len(), 54, "seeded case count");
    let mut diffs = Vec::new();
    let mut compared = 0;
    for path in &paths {
        let case = load_case(path);
        for metric in METRICS {
            let (d, n) = compare_with_python(&case, metric);
            diffs.extend(d);
            compared += n;
        }
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
    eprintln!(
        "seeded: {} cases, {compared} score maps compared with Python",
        paths.len()
    );
}

/// Every CLEAR, Identity, and Count entry of the 20 `reference_benchmarks.json` cases:
/// the nine local benchmarks scored with seeded perturbed predictions (`py-reference`).
#[test]
fn reference_benchmarks_match_python() {
    let Some(root) = real_testdata_dir("CLEAR/Identity/Count reference_benchmarks.json") else {
        return;
    };
    let names = real_case_names(Path::new(&root), "reference_benchmarks");
    let mut cases = 0;
    for name in &names {
        let Some(case) = load_real_case(&root, "reference_benchmarks", name) else {
            continue;
        };
        let (fields, compared) = check_gate_case(&case);
        eprintln!(
            "reference_benchmarks/{name}: {fields} oracle fields, {compared} score maps \
             compared with Python"
        );
        cases += 1;
    }
    eprintln!("reference_benchmarks: {cases} of 20 cases checked");
}
