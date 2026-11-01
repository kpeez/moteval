//! Real-data loader gate: on all nine local benchmarks, every split with GT, the Rust
//! loader and readers give what Python moteval gives, and every class view's
//! `SequenceData` equals Python's bit for bit.
//!
//! Inputs, under `MOTEVAL_REAL_TESTDATA`:
//! - `loaders_real/<benchmark>_<split>/` from `scripts/testdata/dump_loaders.py`: Python's
//!   loaded GT, the seeded prediction files, Python's parse of them, and (masks) the
//!   mask geometry calls;
//! - `reference_benchmarks/<benchmark>_<split>/` from `scripts/testdata/dump_scenarios.py
//!   --real-data`: Python's `SequenceData` for each class view, built from the same
//!   prediction files.
//!
//! The GT comes from `MOTEVAL_DATA_ROOT`. Missing data skips loudly
//! (`SKIPPING REAL-DATA GATE`); a skip never counts as a pass.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use crate::harness::{as_array, as_str, as_u64, assert_f64s_bit_equal, read_json};
use moteval::benchmarks::load_dataset;
use moteval::convert::{MaskGeometry, build_mask_sequence_data_with, build_sequence_data};
use moteval::formats::{MaskTrack, Track, read_mot, read_mots};
use moteval::model::{Geometry, Matrix, Sequence, SequenceData};
use moteval::rle::{RleError, RleMask};
use serde_json::Value as Json;

/// `data.bin` with its array records.
struct Blob {
    data: Vec<u8>,
}

impl Blob {
    fn words(&self, rec: &Json) -> impl Iterator<Item = [u8; 8]> + '_ {
        let offset = usize::try_from(as_u64(&rec["offset"])).expect("usize");
        let len = usize::try_from(as_u64(&rec["len"])).expect("usize");
        let (words, rest) = self.data[offset..offset + 8 * len].as_chunks::<8>();
        assert!(rest.is_empty());
        words.iter().copied()
    }

    fn i64s(&self, rec: &Json) -> Vec<i64> {
        self.words(rec).map(i64::from_le_bytes).collect()
    }

    fn f64s(&self, rec: &Json) -> Vec<f64> {
        self.words(rec).map(f64::from_le_bytes).collect()
    }

    fn bytes(&self, rec: &Json) -> &[u8] {
        let offset = usize::try_from(as_u64(&rec["offset"])).expect("usize");
        let len = usize::try_from(as_u64(&rec["len"])).expect("usize");
        &self.data[offset..offset + len]
    }

    fn box_rows(&self, rec: &Json) -> Vec<Track> {
        let ints = self.i64s(&rec["ints"]);
        let floats = self.f64s(&rec["floats"]);
        let (ints, ints_rest) = ints.as_chunks::<3>();
        let (floats, floats_rest) = floats.as_chunks::<5>();
        let rows = usize::try_from(as_u64(&rec["len"])).expect("usize");
        assert!(ints_rest.is_empty() && floats_rest.is_empty());
        assert_eq!((ints.len(), floats.len()), (rows, rows), "row arrays");
        ints.iter()
            .zip(floats)
            .map(|(i, f)| Track {
                frame: i[0],
                track_id: i[1],
                class_id: i[2],
                x: f[0],
                y: f[1],
                w: f[2],
                h: f[3],
                conf: f[4],
            })
            .collect()
    }

    fn mask_rows(&self, rec: &Json) -> Vec<MaskTrack> {
        let ints = self.i64s(&rec["ints"]);
        let lens = self.i64s(&rec["rle_lens"]);
        let bytes = self.bytes(&rec["rle_bytes"]);
        let mut at = 0;
        let (ints, ints_rest) = ints.as_chunks::<5>();
        let rows = usize::try_from(as_u64(&rec["len"])).expect("usize");
        assert!(ints_rest.is_empty());
        assert_eq!((ints.len(), lens.len()), (rows, rows), "row arrays");
        ints.iter()
            .zip(lens)
            .map(|(i, len)| {
                let len = usize::try_from(len).expect("usize");
                let rle = std::str::from_utf8(&bytes[at..at + len])
                    .expect("ASCII")
                    .to_owned();
                at += len;
                MaskTrack {
                    frame: i[0],
                    track_id: i[1],
                    class_id: i[2],
                    img_h: i[3],
                    img_w: i[4],
                    rle,
                }
            })
            .collect()
    }
}

fn same_tracks(expected: &[Track], got: &[Track], label: &str) {
    assert_eq!(expected.len(), got.len(), "{label}: row count");
    for (k, (e, g)) in expected.iter().zip(got).enumerate() {
        let ints = |t: &Track| (t.frame, t.track_id, t.class_id);
        let bits = |t: &Track| [t.x, t.y, t.w, t.h, t.conf].map(f64::to_bits);
        assert_eq!(ints(e), ints(g), "{label}: row {k}");
        assert_eq!(bits(e), bits(g), "{label}: row {k} floats {e:?} vs {g:?}");
    }
}

fn rle_text(r: &RleMask) -> (usize, usize, Vec<u8>) {
    (r.height(), r.width(), r.to_compressed())
}

/// Every field of two `SequenceData`, floats bit for bit.
fn same_sequence_data(expected: &SequenceData, got: &SequenceData, label: &str) {
    assert_eq!(expected.name(), got.name(), "{label}: name");
    let counts = |d: &SequenceData| {
        (
            d.num_timesteps(),
            d.num_gt_ids(),
            d.num_pred_ids(),
            d.num_gt_dets(),
            d.num_pred_dets(),
        )
    };
    assert_eq!(counts(expected), counts(got), "{label}: counts");
    assert_eq!(expected.gt_ids(), got.gt_ids(), "{label}: gt_ids");
    assert_eq!(expected.pred_ids(), got.pred_ids(), "{label}: pred_ids");
    for t in 0..expected.num_timesteps() {
        let at = format!("{label} t={t}");
        assert_f64s_bit_equal(
            &expected.pred_confidences()[t],
            &got.pred_confidences()[t],
            &format!("{at} pred_confidences"),
        );
        let (e, g) = (&expected.similarity()[t], &got.similarity()[t]);
        assert_eq!(e.shape(), g.shape(), "{at} similarity shape");
        assert_f64s_bit_equal(e.data(), g.data(), &format!("{at} similarity"));
        match (expected.geometry(), got.geometry()) {
            (Geometry::Boxes { gt: eg, pred: ep }, Geometry::Boxes { gt: gg, pred: gp }) => {
                let flat = |b: &[[f64; 4]]| b.iter().flatten().copied().collect::<Vec<_>>();
                assert_f64s_bit_equal(&flat(&eg[t]), &flat(&gg[t]), &format!("{at} gt boxes"));
                assert_f64s_bit_equal(&flat(&ep[t]), &flat(&gp[t]), &format!("{at} pred boxes"));
            }
            (Geometry::Masks { gt: eg, pred: ep }, Geometry::Masks { gt: gg, pred: gp }) => {
                let texts = |m: &[RleMask]| m.iter().map(rle_text).collect::<Vec<_>>();
                assert_eq!(texts(&eg[t]), texts(&gg[t]), "{at} gt masks");
                assert_eq!(texts(&ep[t]), texts(&gp[t]), "{at} pred masks");
            }
            _ => panic!("{label}: geometry kinds differ"),
        }
    }
}

/// FNV-1a 64 over `"{h} {w} {counts}\n"` for each mask (`dump_loaders.mask_digest`).
fn mask_digest(masks: &[RleMask]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for m in masks {
        let mut bytes = format!("{} {} ", m.height(), m.width()).into_bytes();
        bytes.extend(m.to_compressed());
        bytes.push(b'\n');
        for b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn matrix(j: &Json) -> Matrix {
    let size = |k: usize| usize::try_from(as_u64(&j[k])).expect("usize");
    let data = as_array(&j[2])
        .iter()
        .map(|b| f64::from_bits(as_u64(b)))
        .collect();
    Matrix::new(size(0), size(1), data).expect("a matrix")
}

/// The recorded mask geometry calls of one conversion, replayed in order; each checks
/// that Rust passes the masks Python passed (by digest).
struct ReplayGeometry {
    calls: RefCell<VecDeque<Json>>,
    label: String,
}

impl ReplayGeometry {
    fn next(&self, kind: &str, inputs: &[&[RleMask]]) -> Json {
        let call = self
            .calls
            .borrow_mut()
            .pop_front()
            .unwrap_or_else(|| panic!("{}: an extra {kind} call", self.label));
        assert_eq!(as_str(&call["kind"]), kind, "{}: call order", self.label);
        let digests: Vec<u64> = inputs.iter().map(|m| mask_digest(m)).collect();
        let expected: Vec<u64> = as_array(&call["digests"]).iter().map(as_u64).collect();
        assert_eq!(digests, expected, "{}: {kind} inputs", self.label);
        call["output"].clone()
    }
}

impl MaskGeometry for ReplayGeometry {
    fn masks_overlap(&self, masks: &[RleMask]) -> Result<bool, RleError> {
        Ok(self
            .next("masks_overlap", &[masks])
            .as_bool()
            .expect("bool"))
    }

    fn ignore_ioa(&self, preds: &[RleMask], ignore: &[RleMask]) -> Result<Matrix, RleError> {
        let merged = self.next("merge_masks", &[ignore]);
        let size = |k: usize| usize::try_from(as_u64(&merged[k])).expect("usize");
        let merged = RleMask::from_compressed(size(0), size(1), as_str(&merged[2]).as_bytes())?;
        Ok(matrix(&self.next("mask_ioa", &[preds, &[merged]])))
    }

    fn mask_iou(&self, gt: &[RleMask], preds: &[RleMask]) -> Result<Matrix, RleError> {
        Ok(matrix(&self.next("mask_iou", &[gt, preds])))
    }
}

/// The data root the GT loads from, or `None` after the loud skip line.
fn data_root(what: &str) -> Option<PathBuf> {
    match std::env::var_os("MOTEVAL_DATA_ROOT") {
        Some(root) if Path::new(&root).is_dir() => Some(PathBuf::from(root)),
        _ => {
            eprintln!(
                "SKIPPING REAL-DATA GATE: {what}: MOTEVAL_DATA_ROOT is not a directory \
                 (set it to the benchmark root, e.g. /data/tracking/external)"
            );
            None
        }
    }
}

fn check_case(case: &str) {
    let Some(testdata) = crate::harness::real_testdata_dir(case) else {
        return;
    };
    let Some(root) = data_root(case) else {
        return;
    };
    let dir = testdata.join("loaders_real").join(case);
    if !dir.join("header.json").is_file() {
        eprintln!(
            "SKIPPING REAL-DATA GATE: {case}: {} not found - generate it with \
             `scripts/testdata/dump_loaders.py`",
            dir.display()
        );
        return;
    }
    let Some(reference) = crate::harness::load_real_case(&testdata, "reference_benchmarks", case)
    else {
        return;
    };
    let header = read_json(&dir.join("header.json"));
    let blob = Blob {
        data: fs::read(dir.join("data.bin")).expect("read data.bin"),
    };
    let benchmark = as_str(&header["benchmark"]);
    let split = as_str(&header["split"]);
    let dataset = load_dataset(benchmark, Some(&root.join(benchmark)), Some(split))
        .unwrap_or_else(|err| panic!("{case}: {err}"));

    // The dataset and its protocol.
    assert_eq!(dataset.name, as_str(&header["name"]), "{case}: name");
    assert_eq!(
        dataset.split,
        as_str(&header["dataset_split"]),
        "{case}: split"
    );
    let p = &dataset.protocol;
    let hp = &header["protocol"];
    assert_eq!(p.name, as_str(&hp["name"]), "{case}: protocol name");
    assert_eq!(
        (
            p.frame_convention.name.as_str(),
            p.frame_convention.first_frame
        ),
        (
            as_str(&hp["convention"][0]),
            hp["convention"][1].as_i64().expect("i64")
        ),
        "{case}: convention"
    );
    let ints = |j: &Json| {
        as_array(j)
            .iter()
            .map(|v| v.as_i64().expect("i64"))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        p.eval_classes,
        ints(&hp["eval_classes"]),
        "{case}: eval_classes"
    );
    assert_eq!(
        p.distractor_classes,
        ints(&hp["distractor_classes"]),
        "{case}: distractors"
    );
    assert_eq!(
        p.distractor_iou_threshold.to_bits(),
        as_u64(&hp["distractor_iou_threshold"])
    );
    assert_eq!(
        p.ignore_iou_threshold.to_bits(),
        as_u64(&hp["ignore_iou_threshold"])
    );
    assert_eq!(
        p.drop_zero_conf_gt,
        hp["drop_zero_conf_gt"].as_bool().expect("bool")
    );
    assert_eq!(p.matching_fill.to_bits(), as_u64(&hp["matching_fill"]));

    let sequences = as_array(&header["sequences"]);
    assert_eq!(
        dataset.sequences.len(),
        sequences.len(),
        "{case}: sequence count"
    );
    let multi_class = p.eval_classes.len() > 1;
    let mut checked_views = 0;
    for (i, (seq, rec)) in dataset.sequences.iter().zip(sequences).enumerate() {
        let label = format!("{case} {}", seq.name());
        assert_eq!(seq.name(), as_str(&rec["name"]), "{label}: name");
        assert_eq!(
            seq.num_timesteps() as u64,
            as_u64(&rec["num_timesteps"]),
            "{label}: num_timesteps"
        );
        let pred_path = dir.join("pred").join(format!("{}.txt", seq.name()));
        match seq {
            Sequence::Boxes(gt) => {
                same_tracks(
                    &blob.box_rows(&rec["tracks"]),
                    &gt.tracks,
                    &format!("{label} GT"),
                );
                same_tracks(
                    &blob.box_rows(&rec["ignore_regions"]),
                    &gt.ignore_regions,
                    &format!("{label} ignore"),
                );
                let preds = read_mot(&pred_path, multi_class).expect("read predictions");
                same_tracks(
                    &blob.box_rows(&rec["preds"]),
                    &preds,
                    &format!("{label} preds"),
                );
                for (v, &cls) in p.eval_classes.iter().enumerate() {
                    let got = build_sequence_data(gt, &preds, p, cls)
                        .unwrap_or_else(|err| panic!("{label}: {err}"));
                    let expected = &reference.views[v].sequences[i];
                    assert_eq!(reference.views[v].class, Some(cls), "{label}: view class");
                    same_sequence_data(&expected.data, &got, &format!("{label} class {cls}"));
                    checked_views += 1;
                }
            }
            Sequence::Masks(gt) => {
                assert_eq!(blob.mask_rows(&rec["tracks"]), gt.tracks, "{label} GT");
                assert_eq!(
                    blob.mask_rows(&rec["ignore_regions"]),
                    gt.ignore_regions,
                    "{label} ignore"
                );
                let preds = read_mots(&pred_path).expect("read predictions");
                assert_eq!(blob.mask_rows(&rec["preds"]), preds, "{label} preds");
                for (v, &cls) in p.eval_classes.iter().enumerate() {
                    let index = v * dataset.sequences.len() + i;
                    let log = read_json(&dir.join(format!("geometry_{index}.json")));
                    let geometry = ReplayGeometry {
                        calls: RefCell::new(as_array(&log).iter().cloned().collect()),
                        label: format!("{label} class {cls}"),
                    };
                    let got = build_mask_sequence_data_with(&geometry, gt, &preds, p, cls)
                        .unwrap_or_else(|err| panic!("{label}: {err}"));
                    assert!(
                        geometry.calls.borrow().is_empty(),
                        "{label}: missing geometry calls"
                    );
                    let expected = &reference.views[v].sequences[i];
                    same_sequence_data(&expected.data, &got, &format!("{label} class {cls}"));
                    checked_views += 1;
                }
            }
        }
    }
    let expected_views: usize = reference.views.iter().map(|v| v.sequences.len()).sum();
    assert_eq!(
        checked_views, expected_views,
        "{case}: every class view checked"
    );
    eprintln!(
        "{case}: {} sequences, {checked_views} class views equal Python",
        sequences.len()
    );
}

macro_rules! real_cases {
    ($($test:ident => $case:literal,)*) => {
        $(
            #[test]
            fn $test() {
                check_case($case);
            }
        )*
    };
}

real_cases! {
    animaltrack_all => "animaltrack_all",
    animaltrack_train => "animaltrack_train",
    animaltrack_test => "animaltrack_test",
    bft_train => "bft_train",
    bft_val => "bft_val",
    bft_test => "bft_test",
    chimpact_train => "chimpact_train",
    chimpact_val => "chimpact_val",
    chimpact_test => "chimpact_test",
    dancetrack_train => "dancetrack_train",
    dancetrack_val => "dancetrack_val",
    gmot40_test => "gmot40_test",
    gmot40_animal => "gmot40_animal",
    mots20_train => "mots20_train",
    panaf500_train => "panaf500_train",
    panaf500_validation => "panaf500_validation",
    panaf500_test => "panaf500_test",
    sportsmot_train => "sportsmot_train",
    sportsmot_val => "sportsmot_val",
    uavdt_all => "uavdt_all",
}
