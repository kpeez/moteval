use std::panic;

use crate::harness::{
    Case, assert_scores_bit_equal, load_real_case, load_synthetic_case, real_case_names,
    real_testdata_dir, synthetic_case_names,
};
use moteval::metrics::{Scores, Value};
use moteval::model::Geometry;

fn oracle_equals_dumped_moteval(case: &Case) -> usize {
    let mut compared = 0;
    for (section, by_metric) in &case.oracle {
        for (metric, want) in by_metric {
            let ours = match section.as_str() {
                "combined" => {
                    assert_eq!(
                        case.views.len(),
                        1,
                        "{}: combined needs one view",
                        case.case
                    );
                    &case.views[0].moteval_combined
                }
                "class_averaged" => &case.moteval_class_averaged,
                "det_averaged" => &case.moteval_det_averaged,
                other => panic!("unknown oracle section {other}"),
            };
            let got = &ours
                .iter()
                .find(|(name, _)| name == metric)
                .unwrap_or_else(|| panic!("{}: no moteval {metric}", case.case))
                .1;
            assert_scores_bit_equal(
                want,
                got,
                &format!("{}/{} {section}.{metric}", case.fixture, case.case),
            );
            compared += 1;
        }
    }
    compared
}

#[test]
fn harness_loads_one_fixture_end_to_end() {
    let case = load_synthetic_case("synthetic_box", "misses");
    assert_eq!(case.fixture, "synthetic_box");
    assert_eq!(case.procedure, "per_class");
    assert_eq!(case.metrics, ["HOTA", "CLEAR", "Identity", "Count"]);
    assert_eq!(case.views.len(), 1);
    assert_eq!(case.views[0].class, Some(1));
    let seq = &case.views[0].sequences[0];
    assert!(matches!(seq.data.geometry(), Geometry::Boxes { .. }));
    assert_eq!(seq.data.similarity().len(), seq.data.num_timesteps());
    let dets: usize = seq.data.gt_ids().iter().map(Vec::len).sum();
    assert_eq!(dets, seq.data.num_gt_dets());
    assert_eq!(oracle_equals_dumped_moteval(&case), 4);

    let want = &case.oracle_section("combined").expect("combined")[0].1;
    for perturb in [|x: f64| f64::from_bits(x.to_bits() ^ 1), |_: f64| f64::NAN] {
        let mut got = Scores::new();
        for (field, value) in want.iter() {
            let changed = match value {
                Value::Array(values) => {
                    let mut values = values.clone();
                    values[0] = perturb(values[0]);
                    Value::Array(values)
                }
                Value::Float(x) => Value::Float(perturb(*x)),
                Value::Matrix(m) => Value::Matrix(m.clone()),
            };
            got.insert(field, changed);
        }
        let result = panic::catch_unwind(|| assert_scores_bit_equal(want, &got, "perturbed"));
        assert!(result.is_err(), "the comparator accepted a changed value");
    }
    let missing = want.declared(&["HOTA"]);
    let result = panic::catch_unwind(|| assert_scores_bit_equal(want, &missing, "missing"));
    assert!(result.is_err(), "the comparator accepted a missing field");
}

#[test]
fn every_synthetic_case_loads_and_matches_its_oracle() {
    let mut cases = 0;
    let mut sections = 0;
    for fixture in ["synthetic_box", "synthetic_mots", "synthetic_trackmap"] {
        let names = synthetic_case_names(fixture);
        assert!(!names.is_empty(), "no cases for {fixture}");
        for name in names {
            let case = load_synthetic_case(fixture, &name);
            sections += oracle_equals_dumped_moteval(&case);
            cases += 1;
        }
    }
    assert_eq!(cases, 37);
    assert!(sections >= 37);
}

#[test]
fn real_data_cases_load_and_match_their_oracle() {
    let Some(root) = real_testdata_dir("harness real-data cases") else {
        return;
    };
    let mut cases = 0;
    for (fixture, expected) in [("real_data", 3), ("reference_benchmarks", 20)] {
        let names = real_case_names(&root, fixture);
        assert_eq!(
            names.len(),
            expected,
            "{fixture}: case count under {}",
            root.display()
        );
        for name in names {
            let case = load_real_case(&root, fixture, &name).expect("listed case exists");
            oracle_equals_dumped_moteval(&case);
            cases += 1;
        }
    }
    eprintln!("checked {cases} real-data cases");
}
