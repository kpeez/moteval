//! J&F and mask similarity against Python moteval and the TrackEval oracle.
//!
//! - Boundary matching: `_boundary_points` and scipy `KDTree` counts on 1,000 seeded mask
//!   pairs (`rust/testdata/jf_boundary.json`).
//! - 300 seeded random J&F sequences scored by Python moteval, errors included, and its
//!   three combiners over groups of them (`rust/testdata/jf_sequences.json`).
//! - Every `synthetic_mots` case: the J&F oracle entries, Python moteval's raw per-sequence
//!   and combined J&F state, and `mask_iou` against the dumped per-frame similarity.
//! - Real data (`MOTEVAL_RUST_TESTDATA`): the MOTS20 entry of `real_data.json` and the
//!   MOTS20 split of the reference benchmarks, checked the same way.

mod harness;
mod maskgen;

use harness::{
    Case, as_array, as_str, as_u64, as_usize, assert_f64s_bit_equal, assert_field_order,
    assert_scores_bit_equal, check_case_against_oracle, f64s_from_bits, load_real_case,
    load_synthetic_case, read_json, real_testdata_dir, synthetic_case_names, testdata_dir,
};
use maskgen::{boundary_case, fnv1a64_bytes, fnv1a64_ints, jf_sequence};
use moteval::metrics::jf::{Boundary, count_within};
use moteval::metrics::{JAndF, Metric, Scores, Value};
use moteval::model::{Geometry, Matrix, SequenceData, SequenceDataParts};
use moteval::rle::{self, RleMask};
use moteval::similarity::mask_iou;
use serde_json::Value as Json;

fn points_fnv(points: &[(i64, i64)]) -> u64 {
    let flat: Vec<u64> = points
        .iter()
        .flat_map(|&(r, c)| [r as u64, c as u64])
        .collect();
    fnv1a64_ints(&flat)
}

#[test]
fn boundary_points_and_matches_equal_scipy_kdtree_on_1000_pairs() {
    let json = read_json(&testdata_dir().join("jf_boundary.json"));
    let cases = as_array(&json["cases"]);
    assert_eq!(cases.len(), 1000);
    let mut points = 0;
    for case in cases {
        let index = as_usize(&case["index"]);
        let (a, b, r) = boundary_case(index);
        assert_eq!(
            fnv1a64_bytes(&a.bytes),
            as_u64(&case["fnv_a"]),
            "{index}: drift"
        );
        assert_eq!(
            fnv1a64_bytes(&b.bytes),
            as_u64(&case["fnv_b"]),
            "{index}: drift"
        );
        assert_eq!(r, as_u64(&case["r"]) as i64, "{index}: drift");
        let ba = Boundary::of(&rle::encode(&a.bytes, a.h, a.w)).unwrap();
        let bb = Boundary::of(&rle::encode(&b.bytes, b.h, b.w)).unwrap();
        for (side, boundary, want) in [("a", &ba, &case["points_a"]), ("b", &bb, &case["points_b"])]
        {
            assert_eq!(
                boundary.points.len(),
                as_usize(&want["n"]),
                "{index}{side}: count"
            );
            assert_eq!(
                points_fnv(&boundary.points),
                as_u64(&want["fnv"]),
                "{index}{side}: points"
            );
        }
        assert_eq!(
            count_within(&ba.points, &bb, r),
            as_usize(&case["match_a"]),
            "{index}: matches of a"
        );
        assert_eq!(
            count_within(&bb.points, &ba, r),
            as_usize(&case["match_b"]),
            "{index}: matches of b"
        );
        points += ba.points.len() + bb.points.len();
    }
    eprintln!("checked {points} boundary points");
}

/// Dumped scores `[[field, [bits]]]`, in Python's insertion order.
fn dumped_scores(json: &Json) -> Scores {
    let mut scores = Scores::new();
    for pair in as_array(json) {
        let values = f64s_from_bits(&pair[1]);
        assert_eq!(values.len(), 1, "J&F fields are scalars");
        scores.insert(as_str(&pair[0]), Value::Float(values[0]));
    }
    scores
}

fn fuzz_sequence_data(index: usize) -> (SequenceData, u64) {
    let spec = jf_sequence(index);
    let mut flat = Vec::new();
    for side in [&spec.gt, &spec.pred] {
        for frame in side {
            for det in frame {
                flat.extend([
                    det.track as u64,
                    det.mask.h as u64,
                    det.mask.w as u64,
                    fnv1a64_bytes(&det.mask.bytes),
                ]);
            }
        }
    }
    let ids = |side: &[Vec<maskgen::Detection>]| -> Vec<Vec<usize>> {
        side.iter()
            .map(|f| f.iter().map(|d| d.track).collect())
            .collect()
    };
    let masks = |side: &[Vec<maskgen::Detection>]| -> Vec<Vec<RleMask>> {
        side.iter()
            .map(|f| {
                f.iter()
                    .map(|d| rle::encode(&d.mask.bytes, d.mask.h, d.mask.w))
                    .collect()
            })
            .collect()
    };
    let (gt_ids, pred_ids) = (ids(&spec.gt), ids(&spec.pred));
    let similarity = gt_ids
        .iter()
        .zip(&pred_ids)
        .map(|(g, p)| Matrix::zeros(g.len(), p.len()))
        .collect();
    let parts = SequenceDataParts {
        name: format!("seq{index}"),
        num_timesteps: spec.num_timesteps,
        num_gt_ids: spec.num_gt_ids,
        num_pred_ids: spec.num_pred_ids,
        num_gt_dets: gt_ids.iter().map(Vec::len).sum(),
        num_pred_dets: pred_ids.iter().map(Vec::len).sum(),
        pred_confidences: pred_ids.iter().map(|p| vec![1.0; p.len()]).collect(),
        gt_ids,
        pred_ids,
        geometry: Geometry::Masks {
            gt: masks(&spec.gt),
            pred: masks(&spec.pred),
        },
        similarity,
    };
    (SequenceData::new(parts).unwrap(), fnv1a64_ints(&flat))
}

#[test]
fn random_sequences_match_python_jf_scores_errors_and_combiners() {
    let json = read_json(&testdata_dir().join("jf_sequences.json"));
    let sequences = as_array(&json["sequences"]);
    assert_eq!(sequences.len(), 300);
    let metric = JAndF;
    let mut scored: Vec<Option<Scores>> = Vec::with_capacity(sequences.len());
    let (mut ok, mut errors) = (0, 0);
    for record in sequences {
        let index = as_usize(&record["index"]);
        let (data, fnv) = fuzz_sequence_data(index);
        assert_eq!(
            fnv,
            as_u64(&record["inputs_fnv"]),
            "seq{index}: generator drift"
        );
        let got = metric.eval_sequence(&data);
        match (record.get("error"), got) {
            (Some(want), Err(err)) => {
                assert_eq!(err.to_string(), as_str(want), "seq{index}: error text");
                errors += 1;
                scored.push(None);
            }
            (None, Ok(scores)) => {
                let want = dumped_scores(&record["scores"]);
                let label = format!("seq{index}");
                assert_field_order(&want, &scores, &label);
                assert_scores_bit_equal(&want, &scores, &label);
                ok += 1;
                scored.push(Some(scores));
            }
            (want, got) => panic!("seq{index}: Python gave {want:?}, Rust gave {got:?}"),
        }
    }
    let groups = as_array(&json["groups"]);
    for (k, group) in groups.iter().enumerate() {
        let members: Vec<&Scores> = as_array(&group["members"])
            .iter()
            .map(|i| scored[as_usize(i)].as_ref().expect("scored member"))
            .collect();
        let label = format!("group {k}");
        for (key, got) in [
            ("combine_sequences", metric.combine_sequences(&members)),
            (
                "class_averaged",
                metric.combine_classes_class_averaged(&members),
            ),
            (
                "det_averaged",
                metric.combine_classes_det_averaged(&members),
            ),
        ] {
            let want = dumped_scores(&group[key]);
            let got = got.unwrap();
            assert_field_order(&want, &got, &format!("{label} {key}"));
            assert_scores_bit_equal(&want, &got, &format!("{label} {key}"));
        }
    }
    assert_eq!(ok + errors, 300);
    eprintln!(
        "checked {ok} scored sequences, {errors} errors, {} combiner groups",
        groups.len()
    );
}

/// `mask_iou(gt_masks[t], pred_masks[t])` equals the dumped `similarity[t]`, bit for bit.
/// Returns the number of frames compared.
fn check_mask_similarity(case: &Case) -> usize {
    let mut frames = 0;
    for view in &case.views {
        for seq in &view.sequences {
            let data = &seq.data;
            let (gt, pred) = (data.gt_masks().unwrap(), data.pred_masks().unwrap());
            for t in 0..data.num_timesteps() {
                let got = mask_iou(&gt[t], &pred[t]).unwrap();
                let want = &data.similarity()[t];
                assert_eq!(got.shape(), want.shape());
                assert_f64s_bit_equal(
                    want.data(),
                    got.data(),
                    &format!(
                        "{}/{} {} frame {t} similarity",
                        case.fixture, case.case, seq.key
                    ),
                );
                frames += 1;
            }
        }
    }
    frames
}

fn find<'a>(by_metric: &'a [(String, Scores)], metric: &str) -> Option<&'a Scores> {
    by_metric.iter().find(|(m, _)| m == metric).map(|(_, s)| s)
}

/// J&F on one case: the oracle fields, and every J&F result Python moteval dumped
/// (per-sequence, combined, class-averaged, det-averaged), with field order. Returns
/// the number of oracle fields checked.
fn check_jf_case(case: &Case) -> usize {
    let metric = JAndF;
    let label = format!("{}/{}", case.fixture, case.case);
    let checked = check_case_against_oracle(case, &metric);
    let mut combined = Vec::new();
    for view in &case.views {
        let mut per_sequence = Vec::new();
        for (seq, (key, by_metric)) in view.sequences.iter().zip(&view.moteval_per_sequence) {
            assert_eq!(&seq.key, key);
            let got = metric.eval_sequence(&seq.data).unwrap();
            if let Some(want) = find(by_metric, "JAndF") {
                let l = format!("{label} {key}");
                // Real-data dumps hold the declared fields only, in the same order.
                assert_field_order(want, &got.declared(&field_names(want)), &l);
                assert_scores_bit_equal(want, &got, &l);
            }
            per_sequence.push(got);
        }
        let refs: Vec<&Scores> = per_sequence.iter().collect();
        let got = metric.combine_sequences(&refs).unwrap();
        let want = find(&view.moteval_combined, "JAndF").expect("moteval combined J&F");
        assert_field_order(want, &got, &format!("{label} combined"));
        assert_scores_bit_equal(want, &got, &format!("{label} combined"));
        combined.push(got);
    }
    if case.views.len() > 1 {
        let refs: Vec<&Scores> = combined.iter().collect();
        for (want_all, got) in [
            (
                &case.moteval_class_averaged,
                metric.combine_classes_class_averaged(&refs).unwrap(),
            ),
            (
                &case.moteval_det_averaged,
                metric.combine_classes_det_averaged(&refs).unwrap(),
            ),
        ] {
            let want = find(want_all, "JAndF").expect("moteval class J&F");
            assert_field_order(want, &got, &label);
            assert_scores_bit_equal(want, &got, &label);
        }
    }
    checked
}

fn field_names(scores: &Scores) -> Vec<&'static str> {
    scores
        .iter()
        .map(|(field, _)| {
            *moteval::metrics::jf::FIELDS
                .iter()
                .find(|f| **f == field)
                .unwrap_or_else(|| panic!("unknown J&F field {field}"))
        })
        .collect()
}

#[test]
fn synthetic_mots_cases_match_oracle_and_python() {
    let names = synthetic_case_names("synthetic_mots");
    assert_eq!(names.len(), 7);
    let (mut oracle_fields, mut frames) = (0, 0);
    for name in &names {
        let case = load_synthetic_case("synthetic_mots", name);
        frames += check_mask_similarity(&case);
        assert!(
            case.metrics.iter().any(|m| m == "JAndF"),
            "{name}: J&F not scored"
        );
        oracle_fields += check_jf_case(&case);
    }
    // 8 oracle entries (6 single-class cases, class- and det-averaged), 8 fields each.
    assert_eq!(oracle_fields, 64);
    eprintln!("checked {oracle_fields} oracle fields and {frames} similarity frames");
}

#[test]
fn real_data_mots20_matches_oracle_and_python() {
    let Some(root) = real_testdata_dir("J&F on MOTS20") else {
        return;
    };
    let mut cases = 0;
    for (fixture, case_name) in [
        ("real_data", "mots20"),
        ("reference_benchmarks", "mots20_train"),
    ] {
        let Some(case) = load_real_case(&root, fixture, case_name) else {
            continue;
        };
        let frames = check_mask_similarity(&case);
        let fields = check_jf_case(&case);
        eprintln!("{fixture}/{case_name}: {fields} oracle fields, {frames} similarity frames");
        assert!(fields > 0, "{fixture}/{case_name}: no J&F oracle fields");
        cases += 1;
    }
    assert_eq!(cases, 2, "a MOTS20 real-data case is missing");
}
