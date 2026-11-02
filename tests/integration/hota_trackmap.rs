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

fn assert_same_scores(want: &Scores, got: &Scores, label: &str) {
    assert_field_order(want, got, label);
    assert_scores_bit_equal(want, got, label);
}

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

type SectionResult<'a> = (&'static str, Scores, &'a [(String, Scores)]);

#[derive(Default, Debug)]
struct Tally {
    cases: usize,
    oracle_fields: usize,
    python_score_sets: usize,
}

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
