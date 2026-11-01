//! The hota-trackmap gate: HOTA and TrackMAP against the frozen TrackEval oracle and
//! against Python moteval's own dumped scores, bit for bit.
//!
//! The oracle covers the public fields TrackEval reports. Python moteval's dumped scores
//! also cover the per-sequence private state (`_dt_matches_*`, `_num_dt_*`), every
//! combiner's output, TrackMAP's det-averaged combiner (the one intended divergence from
//! TrackEval, so the oracle does not freeze it), and the dict order, which is the export
//! order. The dumped shapes tell a Python scalar (`[]`) from a one-element array (`[1]`),
//! so the comparison also checks that each field has the kind Python gives it.

use crate::harness::{
    Case, CaseView, assert_field_order, assert_scores_bit_equal, load_real_case,
    load_synthetic_case, metric_by_name, real_case_names, real_testdata_dir, synthetic_case_names,
};
use moteval::metrics::{Metric, Scores};

const PORTED: [&str; 2] = ["HOTA", "TrackMAP"];

fn find<'a>(by_metric: &'a [(String, Scores)], metric: &str) -> Option<&'a Scores> {
    by_metric
        .iter()
        .find(|(name, _)| name == metric)
        .map(|(_, scores)| scores)
}

/// Assert Rust's `got` equals Python's dumped `want`: the same fields in the same order,
/// every value bit for bit and of the same shape.
fn assert_same_scores(want: &Scores, got: &Scores, label: &str) {
    assert_field_order(want, got, label);
    assert_scores_bit_equal(want, got, label);
}

/// Score one view's sequences and combine them, as `evaluate` does.
fn run_view(metric: &dyn Metric, view: &CaseView, label: &str) -> (Vec<Scores>, Scores) {
    let per_sequence: Vec<Scores> = view
        .sequences
        .iter()
        .map(|seq| {
            metric
                .eval_sequence(&seq.data)
                .unwrap_or_else(|err| panic!("{label}: {}: {err}", seq.key))
        })
        .collect();
    let refs: Vec<&Scores> = per_sequence.iter().collect();
    let combined = metric
        .combine_sequences(&refs)
        .unwrap_or_else(|err| panic!("{label}: combine_sequences: {err}"));
    (per_sequence, combined)
}

/// One combiner section: its name, the Rust scores, and Python's scores by metric.
type SectionResult<'a> = (&'static str, Scores, &'a [(String, Scores)]);

/// What one case gave the gate.
#[derive(Default, Debug)]
struct Tally {
    cases: usize,
    /// Oracle (TrackEval) fields equal bit for bit.
    oracle_fields: usize,
    /// Python moteval score sets (per-sequence, combined, class combiners) equal bit for
    /// bit, private fields and field order included.
    python_score_sets: usize,
}

/// Run `metric` on `case` end to end, as `evaluate` does, and compare every oracle field
/// and every score set Python moteval dumped: per-sequence scores (raw for synthetic
/// cases, declared fields for real data), each view's combined state, and the class
/// combiners the procedure runs.
///
/// `per_class`: each view's sequences are scored and combined; with one view the result
/// is checked against `oracle.combined`, with several the class combiners run over the
/// views' raw combined state, in class order. `sequences_as_classes`: one view's declared
/// per-sequence scores go to `combine_classes_class_averaged`.
fn check_case(case: &Case, metric: &dyn Metric, raw_per_sequence: bool, tally: &mut Tally) {
    let name = metric.name();
    let mut combined = Vec::new();
    let mut declared_per_sequence = Vec::new();
    for (v, view) in case.views.iter().enumerate() {
        let label = format!("{}/{} view {v} {name}", case.fixture, case.case);
        let (per_sequence, view_combined) = run_view(metric, view, &label);
        assert_eq!(per_sequence.len(), view.moteval_per_sequence.len());
        for ((seq, got), (key, by_metric)) in view
            .sequences
            .iter()
            .zip(&per_sequence)
            .zip(&view.moteval_per_sequence)
        {
            assert_eq!(&seq.key, key, "{label}: sequence order");
            let want = find(by_metric, name).expect("Python per-sequence scores");
            let got = if raw_per_sequence {
                got.clone()
            } else {
                got.declared(metric.fields())
            };
            assert_same_scores(want, &got, &format!("{label} {key}"));
            tally.python_score_sets += 1;
        }
        let want = find(&view.moteval_combined, name).expect("Python combined scores");
        assert_same_scores(want, &view_combined, &format!("{label} combined"));
        tally.python_score_sets += 1;
        declared_per_sequence.extend(per_sequence.iter().map(|s| s.declared(metric.fields())));
        combined.push(view_combined);
    }

    let label = format!("{}/{} {name}", case.fixture, case.case);
    let mut results: Vec<SectionResult<'_>> = Vec::new();
    match case.procedure.as_str() {
        "per_class" if case.views.len() == 1 => {
            results.push(("combined", combined.remove(0), &[]));
        }
        "per_class" => {
            let refs: Vec<&Scores> = combined.iter().collect();
            let class_averaged = metric
                .combine_classes_class_averaged(&refs)
                .unwrap_or_else(|err| panic!("{label}: class_averaged: {err}"));
            let det_averaged = metric
                .combine_classes_det_averaged(&refs)
                .unwrap_or_else(|err| panic!("{label}: det_averaged: {err}"));
            results.push((
                "class_averaged",
                class_averaged,
                &case.moteval_class_averaged,
            ));
            results.push(("det_averaged", det_averaged, &case.moteval_det_averaged));
        }
        "sequences_as_classes" => {
            let refs: Vec<&Scores> = declared_per_sequence.iter().collect();
            let class_averaged = metric
                .combine_classes_class_averaged(&refs)
                .unwrap_or_else(|err| panic!("{label}: class_averaged: {err}"));
            results.push((
                "class_averaged",
                class_averaged,
                &case.moteval_class_averaged,
            ));
        }
        other => panic!("{label}: unknown procedure {other}"),
    }

    for (section, got, python) in &results {
        if *section != "combined" {
            let want = find(python, name).expect("Python class combiner scores");
            assert_same_scores(want, got, &format!("{label} {section}"));
            tally.python_score_sets += 1;
        }
        if let Some(want) = case.oracle_section(section).and_then(|s| find(s, name)) {
            assert_scores_bit_equal(want, got, &format!("{label} oracle {section}"));
            tally.oracle_fields += want.len();
        }
    }
    tally.cases += 1;
}

fn check_fixture(cases: impl Iterator<Item = Case>, raw: bool) -> [(&'static str, Tally); 2] {
    let mut tallies = PORTED.map(|name| (name, Tally::default()));
    for case in cases {
        for (name, tally) in &mut tallies {
            if case.metrics.iter().any(|m| m == name) {
                check_case(&case, metric_by_name(name).as_ref(), raw, tally);
            }
        }
    }
    tallies
}

fn tally<'a>(tallies: &'a [(&str, Tally); 2], metric: &str) -> &'a Tally {
    &tallies
        .iter()
        .find(|(m, _)| *m == metric)
        .expect("metric")
        .1
}

#[test]
fn synthetic_cases_match_the_oracle_and_python_bit_for_bit() {
    let mut all = Vec::new();
    for fixture in ["synthetic_box", "synthetic_mots", "synthetic_trackmap"] {
        let cases = synthetic_case_names(fixture)
            .into_iter()
            .map(|name| load_synthetic_case(fixture, &name));
        let tallies = check_fixture(cases, true);
        eprintln!("{fixture}: {tallies:?}");
        all.push((fixture, tallies));
    }
    let get = |fixture: &str, metric: &str| {
        let (_, tallies) = all.iter().find(|(f, _)| *f == fixture).expect("fixture");
        let t = tally(tallies, metric);
        assert!(
            t.oracle_fields > 0,
            "{fixture} {metric}: no oracle field checked"
        );
        assert!(
            t.python_score_sets > 0,
            "{fixture} {metric}: no Python scores checked"
        );
        t.cases
    };
    // Every HOTA entry of the box and MOTS fixtures, and every TrackMAP entry.
    assert_eq!(get("synthetic_box", "HOTA"), 18);
    assert_eq!(get("synthetic_mots", "HOTA"), 7);
    assert_eq!(get("synthetic_box", "TrackMAP"), 1);
    assert_eq!(get("synthetic_trackmap", "TrackMAP"), 12);
}

#[test]
fn real_data_hota_matches_the_oracle_and_python() {
    let Some(root) = real_testdata_dir("hota-trackmap real_data.json") else {
        return;
    };
    let names = real_case_names(&root, "real_data");
    assert_eq!(names, ["dancetrack_val", "mots20", "sportsmot_val"]);
    let cases = names
        .iter()
        .map(|name| load_real_case(&root, "real_data", name).expect("listed case"));
    let tallies = check_fixture(cases, false);
    eprintln!("real_data: {tallies:?}");
    let hota = tally(&tallies, "HOTA");
    assert_eq!(hota.cases, 3, "real_data HOTA cases");
    assert_eq!(hota.oracle_fields, 3 * 15, "real_data HOTA oracle fields");
}

/// Not a listed package gate (integration runs the full reference set), but it is the
/// only real-data test of TrackMAP: long and sparse tracks stress the set-order replay.
#[test]
fn reference_benchmarks_hota_and_trackmap_match_python() {
    let Some(root) = real_testdata_dir("hota-trackmap reference_benchmarks.json") else {
        return;
    };
    let names = real_case_names(&root, "reference_benchmarks");
    assert_eq!(names.len(), 20, "reference_benchmarks cases");
    let cases = names
        .iter()
        .map(|name| load_real_case(&root, "reference_benchmarks", name).expect("listed case"));
    let tallies = check_fixture(cases, false);
    eprintln!("reference_benchmarks: {tallies:?}");
    assert_eq!(tally(&tallies, "HOTA").cases, 20, "reference HOTA cases");
    assert_eq!(
        tally(&tallies, "TrackMAP").cases,
        19,
        "reference TrackMAP cases"
    );
}
