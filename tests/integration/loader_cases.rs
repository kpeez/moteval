//! Every call `tests/test_loaders.py` and `tests/test_data.py` make into the ported code,
//! replayed in Rust: the same data, or the same error class and message
//! (`tests/integration/data/loader_cases/`, recorded by `scripts/testdata/record_loader_cases.py`).
//!
//! Each case restores the files the test had written before the call into a scratch
//! directory that stands for `<TMP>`, runs the Rust function, and compares its output in
//! the recorder's encoding: floats as bits, so the comparison is exact.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::harness::{as_array, as_str, as_u64, read_json, testdata_dir};
use moteval::benchmarks::motchallenge::{
    MOTChallengeConfig, load_layout, load_motchallenge, max_frame_seq_length, read_seq_length,
};
use moteval::benchmarks::{
    BENCHMARKS, BenchmarkError, default_data_root_with_env, load_dataset_with_env, mots20,
};
use moteval::convert::{
    ConvertError, MaskGeometry, build_mask_sequence_data_with, build_sequence_data,
};
use moteval::eval::EvalError;
use moteval::formats::pytext::py_str;
use moteval::formats::{
    FormatError, MaskTrack, Track, missing_prediction_files, read_mot, read_mots, write_mot,
    write_mots,
};
use moteval::model::{
    FrameConvention, Geometry, GtSequence, MOTDataset, MaskGtSequence, Matrix, Protocol, Sequence,
    SequenceData,
};
use moteval::protocol::{PreprocessedFrame, RawFrame, preprocess_frame};
use moteval::rle::{RleError, RleMask};
use moteval::similarity::{box_ioa, box_iou};
use serde_json::{Value as Json, json};

// ----------------------------------------------------------------------- decoding

fn int(j: &Json) -> i64 {
    j.as_i64()
        .unwrap_or_else(|| panic!("expected an integer, got {j}"))
}

fn float(j: &Json) -> f64 {
    f64::from_bits(as_u64(j))
}

fn track(j: &Json) -> Track {
    let v = as_array(j);
    Track {
        frame: int(&v[0]),
        track_id: int(&v[1]),
        class_id: int(&v[2]),
        x: float(&v[3]),
        y: float(&v[4]),
        w: float(&v[5]),
        h: float(&v[6]),
        conf: float(&v[7]),
    }
}

fn mask_track(j: &Json) -> MaskTrack {
    let v = as_array(j);
    MaskTrack {
        frame: int(&v[0]),
        track_id: int(&v[1]),
        class_id: int(&v[2]),
        img_h: int(&v[3]),
        img_w: int(&v[4]),
        rle: as_str(&v[5]).to_owned(),
    }
}

fn protocol(j: &Json) -> Protocol {
    let ints = |k: &str| as_array(&j[k]).iter().map(int).collect::<Vec<_>>();
    Protocol {
        name: as_str(&j["name"]).to_owned(),
        frame_convention: FrameConvention::new(
            as_str(&j["convention"][0]),
            int(&j["convention"][1]),
        ),
        eval_classes: ints("eval_classes"),
        distractor_classes: ints("distractor_classes"),
        distractor_iou_threshold: float(&j["distractor_iou_threshold"]),
        ignore_iou_threshold: float(&j["ignore_iou_threshold"]),
        drop_zero_conf_gt: j["drop_zero_conf_gt"].as_bool().expect("a bool"),
        matching_fill: float(&j["matching_fill"]),
    }
}

fn sequence(j: &Json) -> Sequence {
    let name = as_str(&j["name"]).to_owned();
    let num_timesteps = usize::try_from(as_u64(&j["num_timesteps"])).expect("usize");
    if as_str(&j["kind"]) == "masks" {
        let rows = |k: &str| as_array(&j[k]).iter().map(mask_track).collect();
        Sequence::Masks(MaskGtSequence {
            name,
            num_timesteps,
            tracks: rows("tracks"),
            ignore_regions: rows("ignore_regions"),
        })
    } else {
        let rows = |k: &str| as_array(&j[k]).iter().map(track).collect();
        Sequence::Boxes(GtSequence {
            name,
            num_timesteps,
            tracks: rows("tracks"),
            ignore_regions: rows("ignore_regions"),
        })
    }
}

fn rle(j: &Json) -> RleMask {
    let v = as_array(j);
    let size = |k: usize| usize::try_from(as_u64(&v[k])).expect("usize");
    RleMask::from_compressed(size(0), size(1), as_str(&v[2]).as_bytes()).expect("an RLE")
}

fn boxes(j: &Json) -> Vec<[f64; 4]> {
    as_array(&j["boxes"])
        .iter()
        .map(|b| {
            let b = as_array(b);
            [float(&b[0]), float(&b[1]), float(&b[2]), float(&b[3])]
        })
        .collect()
}

fn matrix(j: &Json) -> Matrix {
    let rows = usize::try_from(as_u64(&j[0])).expect("usize");
    let cols = usize::try_from(as_u64(&j[1])).expect("usize");
    let data = as_array(&j[2]).iter().map(float).collect();
    Matrix::new(rows, cols, data).expect("a matrix")
}

fn ints(j: &Json) -> Vec<i64> {
    as_array(j).iter().map(int).collect()
}

fn floats(j: &Json) -> Vec<f64> {
    as_array(j).iter().map(float).collect()
}

// ----------------------------------------------------------------------- encoding

fn enc_track(t: &Track) -> Json {
    json!([
        t.frame,
        t.track_id,
        t.class_id,
        t.x.to_bits(),
        t.y.to_bits(),
        t.w.to_bits(),
        t.h.to_bits(),
        t.conf.to_bits()
    ])
}

fn enc_mask_track(t: &MaskTrack) -> Json {
    json!([t.frame, t.track_id, t.class_id, t.img_h, t.img_w, t.rle])
}

fn enc_protocol(p: &Protocol) -> Json {
    json!({
        "name": p.name,
        "convention": [p.frame_convention.name, p.frame_convention.first_frame],
        "eval_classes": p.eval_classes,
        "distractor_classes": p.distractor_classes,
        "distractor_iou_threshold": p.distractor_iou_threshold.to_bits(),
        "ignore_iou_threshold": p.ignore_iou_threshold.to_bits(),
        "drop_zero_conf_gt": p.drop_zero_conf_gt,
        "matching_fill": p.matching_fill.to_bits(),
    })
}

fn enc_sequence(s: &Sequence) -> Json {
    match s {
        Sequence::Boxes(s) => json!({
            "kind": "boxes",
            "name": s.name,
            "num_timesteps": s.num_timesteps,
            "tracks": s.tracks.iter().map(enc_track).collect::<Vec<_>>(),
            "ignore_regions": s.ignore_regions.iter().map(enc_track).collect::<Vec<_>>(),
        }),
        Sequence::Masks(s) => json!({
            "kind": "masks",
            "name": s.name,
            "num_timesteps": s.num_timesteps,
            "tracks": s.tracks.iter().map(enc_mask_track).collect::<Vec<_>>(),
            "ignore_regions": s.ignore_regions.iter().map(enc_mask_track).collect::<Vec<_>>(),
        }),
    }
}

fn enc_dataset(d: &MOTDataset) -> Json {
    json!({
        "name": d.name,
        "split": d.split,
        "protocol": enc_protocol(&d.protocol),
        "sequences": d.sequences.iter().map(enc_sequence).collect::<Vec<_>>(),
    })
}

fn enc_matrix(m: &Matrix) -> Json {
    let data: Vec<u64> = m.data().iter().map(|v| v.to_bits()).collect();
    json!([m.rows(), m.cols(), data])
}

fn enc_rle(r: &RleMask) -> Json {
    let counts = String::from_utf8(r.to_compressed()).expect("ASCII counts");
    json!([r.height(), r.width(), counts])
}

fn enc_boxes(b: &[[f64; 4]]) -> Json {
    let rows: Vec<Vec<u64>> = b
        .iter()
        .map(|r| r.iter().map(|v| v.to_bits()).collect())
        .collect();
    json!({"boxes": rows})
}

fn enc_masks(m: &[RleMask]) -> Json {
    json!({"masks": m.iter().map(enc_rle).collect::<Vec<_>>()})
}

fn enc_bits(v: &[f64]) -> Json {
    json!(v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
}

fn enc_sequence_data(d: &SequenceData) -> Json {
    let (gt_dets, pred_dets): (Vec<Json>, Vec<Json>) = match d.geometry() {
        Geometry::Boxes { gt, pred } => (
            gt.iter().map(|f| enc_boxes(f)).collect(),
            pred.iter().map(|f| enc_boxes(f)).collect(),
        ),
        Geometry::Masks { gt, pred } => (
            gt.iter().map(|f| enc_masks(f)).collect(),
            pred.iter().map(|f| enc_masks(f)).collect(),
        ),
    };
    json!({
        "name": d.name(),
        "num_timesteps": d.num_timesteps(),
        "num_gt_ids": d.num_gt_ids(),
        "num_pred_ids": d.num_pred_ids(),
        "num_gt_dets": d.num_gt_dets(),
        "num_pred_dets": d.num_pred_dets(),
        "gt_ids": d.gt_ids(),
        "pred_ids": d.pred_ids(),
        "pred_confidences": d.pred_confidences().iter().map(|f| enc_bits(f)).collect::<Vec<_>>(),
        "gt_dets": gt_dets,
        "pred_dets": pred_dets,
        "similarity": d.similarity().iter().map(enc_matrix).collect::<Vec<_>>(),
    })
}

fn enc_preprocessed(f: &PreprocessedFrame<[f64; 4]>) -> Json {
    json!({
        "gt_ids": f.gt_ids,
        "gt_dets": enc_boxes(&f.gt_dets),
        "pred_ids": f.pred_ids,
        "pred_confidences": enc_bits(&f.pred_confidences),
        "pred_dets": enc_boxes(&f.pred_dets),
        "similarity": enc_matrix(&f.similarity),
    })
}

// ----------------------------------------------------------------------- the replay

/// A failed Rust call: its Python exception class and its message.
struct Failure {
    class: &'static str,
    message: String,
}

impl From<FormatError> for Failure {
    fn from(err: FormatError) -> Self {
        Self {
            class: err.python_exception(),
            message: err.to_string(),
        }
    }
}

impl From<BenchmarkError> for Failure {
    fn from(err: BenchmarkError) -> Self {
        Self {
            class: err.python_exception(),
            message: err.to_string(),
        }
    }
}

impl From<ConvertError> for Failure {
    fn from(err: ConvertError) -> Self {
        Self {
            class: err.python_exception(),
            message: err.to_string(),
        }
    }
}

impl From<moteval::lsap::LsapError> for Failure {
    fn from(err: moteval::lsap::LsapError) -> Self {
        Self {
            class: "ValueError",
            message: err.to_string(),
        }
    }
}

/// The Python class a recorded exception counts as: every `OSError` subclass is
/// `OSError`, every configparser interpolation error is `InterpolationError`.
fn python_class(name: &str) -> &str {
    match name {
        "FileNotFoundError" | "IsADirectoryError" | "NotADirectoryError" | "PermissionError" => {
            "OSError"
        }
        n if n.starts_with("Interpolation") => "InterpolationError",
        n => n,
    }
}

/// The recorded geometry calls of one mask conversion, replayed in order. Each call
/// checks that Rust passes the same masks Python passed.
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
            .unwrap_or_else(|| panic!("{}: Rust made an extra {kind} call", self.label));
        assert_eq!(as_str(&call["kind"]), kind, "{}: call order", self.label);
        let got: Vec<Json> = inputs
            .iter()
            .map(|m| json!(m.iter().map(enc_rle).collect::<Vec<_>>()))
            .collect();
        assert_eq!(
            &json!(got),
            &call["inputs"],
            "{}: {kind} inputs",
            self.label
        );
        call["output"].clone()
    }
}

impl MaskGeometry for ReplayGeometry {
    fn masks_overlap(&self, masks: &[RleMask]) -> Result<bool, RleError> {
        Ok(self
            .next("masks_overlap", &[masks])
            .as_bool()
            .expect("a bool"))
    }

    fn ignore_ioa(&self, preds: &[RleMask], ignore: &[RleMask]) -> Result<Matrix, RleError> {
        let merged = rle(&self.next("merge_masks", &[ignore]));
        Ok(matrix(&self.next("mask_ioa", &[preds, &[merged]])))
    }

    fn mask_iou(&self, gt: &[RleMask], preds: &[RleMask]) -> Result<Matrix, RleError> {
        Ok(matrix(&self.next("mask_iou", &[gt, preds])))
    }
}

/// Restore a snapshot into a fresh `dir`.
fn materialize(snapshot: &Json, dir: &Path) {
    if dir.exists() {
        fs::remove_dir_all(dir).expect("remove the old case dir");
    }
    fs::create_dir_all(dir).expect("create the case dir");
    for entry in as_array(snapshot) {
        let path = dir.join(as_str(&entry["path"]));
        if entry.get("dir").is_some() {
            fs::create_dir_all(&path).expect("create a dir");
        } else if let Some(text) = entry.get("text") {
            fs::write(&path, as_str(text)).expect("write a file");
        } else {
            let hex = as_str(&entry["hex"]);
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
                .collect();
            fs::write(&path, bytes).expect("write a file");
        }
    }
}

struct Replay<'a> {
    tmp: &'a str,
    label: String,
}

impl Replay<'_> {
    fn path(&self, j: &Json) -> PathBuf {
        PathBuf::from(as_str(j).replace("<TMP>", self.tmp))
    }

    fn env(&self, j: &Json) -> Option<OsString> {
        j.as_str()
            .map(|v| OsString::from(v.replace("<TMP>", self.tmp)))
    }

    /// Run one case; the output in the recorder's encoding, or the failure.
    fn run(&self, kind: &str, inputs: &Json, case: &Json) -> Result<Json, Failure> {
        let split = |j: &Json| as_str(j).to_owned();
        Ok(match kind {
            "read_mot" => {
                let class_column = inputs["class_column"].as_bool().expect("a bool");
                let tracks = read_mot(&self.path(&inputs["path"]), class_column)?;
                json!(tracks.iter().map(enc_track).collect::<Vec<_>>())
            }
            "read_mots" => {
                let tracks = read_mots(&self.path(&inputs["path"]))?;
                json!(tracks.iter().map(enc_mask_track).collect::<Vec<_>>())
            }
            "write_mot" | "write_mots" => {
                let path = self.path(&inputs["path"]);
                if kind == "write_mot" {
                    let tracks: Vec<Track> =
                        as_array(&inputs["tracks"]).iter().map(track).collect();
                    write_mot(&path, &tracks)?;
                } else {
                    let tracks: Vec<MaskTrack> =
                        as_array(&inputs["tracks"]).iter().map(mask_track).collect();
                    write_mots(&path, &tracks)?;
                }
                json!(fs::read_to_string(&path).expect("the written file"))
            }
            "build_sequence_data" => {
                let Sequence::Boxes(gt) = sequence(&inputs["gt"]) else {
                    panic!("box GT")
                };
                let preds: Vec<Track> =
                    as_array(&inputs["pred_tracks"]).iter().map(track).collect();
                let data = build_sequence_data(
                    &gt,
                    &preds,
                    &protocol(&inputs["protocol"]),
                    int(&inputs["cls_id"]),
                )?;
                enc_sequence_data(&data)
            }
            "build_mask_sequence_data" => {
                let Sequence::Masks(gt) = sequence(&inputs["gt"]) else {
                    panic!("mask GT")
                };
                let preds: Vec<MaskTrack> = as_array(&inputs["pred_tracks"])
                    .iter()
                    .map(mask_track)
                    .collect();
                let geometry = ReplayGeometry {
                    calls: RefCell::new(as_array(&case["geometry"]).iter().cloned().collect()),
                    label: self.label.clone(),
                };
                let data = build_mask_sequence_data_with(
                    &geometry,
                    &gt,
                    &preds,
                    &protocol(&inputs["protocol"]),
                    int(&inputs["cls_id"]),
                )?;
                assert!(
                    geometry.calls.borrow().is_empty(),
                    "{}: Rust made fewer geometry calls than Python",
                    self.label
                );
                enc_sequence_data(&data)
            }
            "preprocess_frame" => {
                let f = &inputs["frame"];
                let frame = RawFrame {
                    gt_ids: ints(&f["gt_ids"]),
                    gt_classes: ints(&f["gt_classes"]),
                    gt_conf: floats(&f["gt_conf"]),
                    gt_dets: boxes(&f["gt_dets"]),
                    pred_ids: ints(&f["pred_ids"]),
                    pred_classes: ints(&f["pred_classes"]),
                    pred_confidences: floats(&f["pred_confidences"]),
                    pred_dets: boxes(&f["pred_dets"]),
                    similarity: matrix(&f["similarity"]),
                    ignore_ioa: matrix(&f["ignore_ioa"]),
                };
                let out = preprocess_frame(
                    &frame,
                    &protocol(&inputs["protocol"]),
                    int(&inputs["cls_id"]),
                )?;
                enc_preprocessed(&out)
            }
            "box_iou" => enc_matrix(&box_iou(&boxes(&inputs["a"]), &boxes(&inputs["b"]))),
            "box_ioa" => enc_matrix(&box_ioa(&boxes(&inputs["a"]), &boxes(&inputs["b"]))),
            "default_data_root" => json!(py_str(&default_data_root_with_env(
                self.env(&inputs["env"])
            )?)),
            "load_dataset" => {
                let root = inputs["root"].as_str().map(|_| self.path(&inputs["root"]));
                let dataset = load_dataset_with_env(
                    as_str(&inputs["name"]),
                    root.as_deref(),
                    inputs["split"].as_str(),
                    self.env(&inputs["env"]),
                )?;
                enc_dataset(&dataset)
            }
            "load_layout" => {
                let c = &inputs["config"];
                let config = MOTChallengeConfig {
                    class_id: int(&c["class_id"]),
                    ..MOTChallengeConfig::new(as_str(&c["name"]), protocol(&c["protocol"]))
                };
                enc_dataset(&load_layout(
                    &config,
                    &self.path(&inputs["root"]),
                    as_str(&inputs["split"]),
                )?)
            }
            "load_motchallenge" => enc_dataset(&load_motchallenge(
                &self.path(&inputs["root"]),
                as_str(&inputs["split"]),
            )?),
            "load_mots20" => enc_dataset(&mots20::load_mots20(
                &self.path(&inputs["root"]),
                as_str(&inputs["split"]),
            )?),
            "load_mots" => enc_dataset(&mots20::load_mots(
                &self.path(&inputs["root"]),
                as_str(&inputs["split"]),
                int(&inputs["class_id"]),
            )?),
            "read_seq_length" => json!(read_seq_length(&self.path(&inputs["seq_dir"]))?),
            "max_frame_seq_length" => {
                let tracks: Vec<Track> = as_array(&inputs["tracks"]).iter().map(track).collect();
                json!(max_frame_seq_length(
                    as_str(&inputs["seq_name"]),
                    &tracks,
                    int(&inputs["offset"])
                )?)
            }
            loader if loader.starts_with("load_") => {
                let name = loader.trim_start_matches("load_");
                let (_, load, _) = BENCHMARKS
                    .iter()
                    .find(|(n, _, _)| *n == name)
                    .unwrap_or_else(|| panic!("no loader {name}"));
                enc_dataset(&load(
                    &self.path(&inputs["root"]),
                    &split(&inputs["split"]),
                )?)
            }
            other => panic!("unknown case kind {other}"),
        })
    }
}

/// `evaluate`'s missing-prediction check: the names, and `EvalError`'s text.
fn check_missing(replay: &Replay<'_>, case: &Json) {
    let inputs = &case["inputs"];
    let pred_dir = replay.path(&inputs["pred_dir"]);
    let names: Vec<&str> = as_array(&inputs["names"]).iter().map(as_str).collect();
    let missing = missing_prediction_files(&pred_dir, names);
    assert_eq!(json!(missing), case["ok"], "{}", replay.label);
    if let Some(expected) = case.get("message") {
        let err = EvalError::MissingPredictions {
            dir: py_str(&pred_dir),
            missing,
        };
        let message = err.to_string().replace(replay.tmp, "<TMP>");
        assert_eq!(message, as_str(expected), "{}", replay.label);
    }
}

#[test]
fn every_recorded_call_replays_bit_for_bit() {
    let root = testdata_dir().join("loader_cases");
    let scratch = std::env::temp_dir().join(format!("moteval-loader-cases-{}", std::process::id()));
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in fs::read_dir(&root).expect("loader_cases exists") {
        for file in fs::read_dir(dir.expect("entry").path()).expect("a test file dir") {
            files.push(file.expect("entry").path());
        }
    }
    files.sort();
    assert_eq!(
        files.len(),
        95,
        "one file per test in test_loaders.py and test_data.py"
    );
    let mut replayed = 0;
    for file in &files {
        let body = read_json(file);
        let test = as_str(&body["test"]);
        for (index, case) in as_array(&body["cases"]).iter().enumerate() {
            let kind = as_str(&case["kind"]);
            let label = format!("{test} case {index} ({kind})");
            let dir = scratch.join("tmp");
            materialize(
                case["snapshot"]
                    .as_str()
                    .map_or(&json!([]), |id| &body["snapshots"][id]),
                &dir,
            );
            let tmp = dir.to_str().expect("a UTF-8 scratch path");
            let replay = Replay {
                tmp,
                label: label.clone(),
            };
            if kind == "missing_predictions" {
                check_missing(&replay, case);
                replayed += 1;
                continue;
            }
            match (replay.run(kind, &case["inputs"], case), case.get("ok")) {
                (Ok(got), Some(expected)) => assert_eq!(&got, expected, "{label}"),
                (Ok(got), None) => panic!(
                    "{label}: Python raises {}, Rust returned {got}",
                    case["error"]
                ),
                (Err(err), Some(_)) => panic!("{label}: Rust failed: {}", err.message),
                (Err(err), None) => {
                    let expected = &case["error"];
                    assert_eq!(
                        err.class,
                        python_class(as_str(&expected["type"])),
                        "{label}: {}",
                        err.message
                    );
                    assert_eq!(
                        err.message.replace(tmp, "<TMP>"),
                        as_str(&expected["message"]),
                        "{label}"
                    );
                }
            }
            replayed += 1;
        }
    }
    fs::remove_dir_all(&scratch).expect("remove the scratch dir");
    assert!(replayed >= 219, "replayed {replayed} cases");
}
