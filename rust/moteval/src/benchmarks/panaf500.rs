//! PanAf500 loader, ported from `moteval/benchmarks/panaf500.py`.
//!
//! One JSON file per video, `<root>/annotations/<split>/<video_id>.json`, shaped
//! `{"annotations": [{"frame_id": int, "detections": [{"bbox": [x1, y1, x2, y2],
//! "ape_id": int}]}]}`. Boxes are converted from `xyxy` to `xywh` (`x2 - x1` is exact on
//! integers, as in Python). Frames are 1-indexed; the sequence length is the largest
//! `frame_id`. Splits are the directories on disk.
//!
//! The loader runs Python's steps in Python's order, so every error Python raises fires
//! first; only then are the values converted to `Track` fields.

use std::path::Path;

use crate::benchmarks::pyjson::{Json, PyNum};
use crate::benchmarks::{BenchmarkError, glob_suffix, json_f64, json_i64, read_json};
use crate::formats::Track;
use crate::formats::pytext::{py_join, py_name, py_stem, py_str};
use crate::model::{FrameConvention, GtSequence, MOTDataset, Protocol, Sequence};

/// The split `load_panaf500` reads when the caller names none.
pub const DEFAULT_SPLIT: &str = "validation";

const CLASS_ID: i64 = 1;

/// One detection as Python holds it before numpy converts it.
struct RawTrack {
    frame: Json,
    track_id: Json,
    x: Json,
    y: Json,
    w: PyNum,
    h: PyNum,
}

fn load_sequence(ann_path: &Path) -> Result<GtSequence, BenchmarkError> {
    let name = py_stem(&py_name(ann_path)).to_owned();
    let data = read_json(ann_path)?;
    let annotations = data.get("annotations")?;
    let mut raw = Vec::new();
    for frame_ann in annotations.iter()? {
        let frame = frame_ann.get("frame_id")?.clone();
        for det in frame_ann.get("detections")?.iter()? {
            // _xyxy_to_xywh(det["bbox"]): x1, y1, x2 - x1, y2 - y1
            let [x1, y1, x2, y2] = det.get("bbox")?.unpack4()?;
            let w = x2.sub(&x1)?;
            let h = y2.sub(&y1)?;
            let track_id = det.get("ape_id")?.clone();
            raw.push(RawTrack {
                frame: frame.clone(),
                track_id,
                x: x1,
                y: y1,
                w,
                h,
            });
        }
    }
    if !data.get("annotations")?.truthy() {
        return Err(BenchmarkError::EmptyGt(name));
    }
    // max(a["frame_id"] for a in data["annotations"])
    let mut num_timesteps: Option<i64> = None;
    for frame_ann in annotations.iter()? {
        let frame = json_i64(frame_ann.get("frame_id")?, "frame_id", ann_path)?;
        num_timesteps = Some(num_timesteps.map_or(frame, |m| m.max(frame)));
    }
    let num_timesteps = num_timesteps.expect("annotations is not empty");
    let num_timesteps = usize::try_from(num_timesteps).map_err(|_| {
        BenchmarkError::Unsupported(format!(
            "{}: the largest frame_id {num_timesteps} is not a sequence length",
            py_str(ann_path)
        ))
    })?;
    let tracks = raw
        .iter()
        .map(|t| {
            Ok(Track {
                frame: json_i64(&t.frame, "frame_id", ann_path)?,
                track_id: json_i64(&t.track_id, "ape_id", ann_path)?,
                x: json_f64(&t.x, "bbox x1", ann_path)?,
                y: json_f64(&t.y, "bbox y1", ann_path)?,
                w: t.w.to_f64(),
                h: t.h.to_f64(),
                conf: 1.0,
                class_id: CLASS_ID,
            })
        })
        .collect::<Result<Vec<_>, BenchmarkError>>()?;
    Ok(GtSequence {
        name,
        num_timesteps,
        tracks,
        ignore_regions: Vec::new(),
    })
}

/// `load_panaf500(root, split)`.
pub fn load_panaf500(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let ann_dir = py_join(&py_join(root, "annotations"), split);
    if !ann_dir.is_dir() {
        return Err(BenchmarkError::SplitDirNotFound(py_str(&ann_dir)));
    }
    let mut names = glob_suffix(&ann_dir, ".json")?;
    names.sort();
    let sequences = names
        .iter()
        .map(|name| load_sequence(&py_join(&ann_dir, name)).map(Sequence::Boxes))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MOTDataset {
        name: "panaf500".to_owned(),
        split: split.to_owned(),
        sequences,
        protocol: Protocol::new("panaf500", FrameConvention::new("1-indexed", 1), vec![1]),
    })
}
