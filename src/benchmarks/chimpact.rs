use std::collections::BTreeMap;
use std::path::Path;

use crate::benchmarks::pyjson::{Json, PyKey};
use crate::benchmarks::{BenchmarkError, PyException, glob_suffix, json_f64, json_i64, read_json};
use crate::formats::Track;
use crate::formats::pytext::{PyInt, int_error_message, py_int, py_join, py_name, py_stem, py_str};
use crate::model::{FrameConvention, GtSequence, MOTDataset, Protocol, Sequence};

pub const DEFAULT_SPLIT: &str = "val";

const KEYFRAME_STRIDE: i64 = 10;
const UNNAMED_BBOX_ID: i128 = 23;
const CLASS_ID: i64 = 1;

// Verbatim from ChimpACT's official `tools/create_coco_format.py`.
pub const VAL_CLIPS: [&str; 17] = [
    "Azibo_ObsChimp_2015_11_25_d_clip_23000_24000",
    "Azibo_ObsChimp_2015_11_26_a_clip_1000_2000",
    "Azibo_ObsChimp_2016_08_02_c_clip_32000_33000",
    "Azibo_ObsChimp_2017_02_27_a_clip_13000_14000",
    "Azibo_ObsChimp_2017_11_10_clip_7000_8000",
    "Azibo_ObsChimp_2017_11_10_clip_8000_9000",
    "Azibo_ObsChimp_2017_06_22_c_clip_46000_47000",
    "Azibo_ObsChimp_2017_06_22_c_clip_67000_68000",
    "Azibo_ObsChimp_2018_07_11_c_clip_0_1000",
    "Azibo_ObsChimp_2018_07_11_c_clip_1000_2000",
    "Azibo_ObsChimp_2018_07_11_c_clip_3000_4000",
    "Azibo_ObsChimp_2018_07_11_c_clip_6000_7000",
    "Azibo_ObsChimp_2018_07_11_c_clip_17000_18000",
    "Azibo_ObsChimp_2018_07_11_c_clip_18000_19000",
    "Azibo_ObsChimp_2018_08_06_a_clip_7000_8000",
    "Azibo_ObsNatascha_2018_06_29_a_clip_15000_16000",
    "Azibo_ObsNatascha_2018_06_29_a_clip_16000_17000",
];

// Verbatim from ChimpACT's official `tools/create_coco_format.py`.
pub const TEST_CLIPS: [&str; 19] = [
    "Azibo_ObsChimp_2015_11_25_d_clip_1000_2000",
    "Azibo_ObsChimp_2015_11_26_a_clip_0_1000",
    "Azibo_ObsChimp_2015_11_26_a_clip_2000_3000",
    "Azibo_ObsChimp_2016_08_02_c_clip_33000_34000",
    "Azibo_ObsChimp_2016_08_15_b_clip_2000_3000",
    "Azibo_ObsChimp_2016_10_27_c_clip_0_1000",
    "Azibo_ObsChimp_2017_02_27_a_clip_14000_15000",
    "Azibo_ObsChimp_2017_11_10_clip_6000_7000",
    "Azibo_ObsChimp_2017_06_22_c_clip_44000_45000",
    "Azibo_ObsChimp_2017_06_22_c_clip_68000_69000",
    "Azibo_ObsChimp_2018_07_06_d_clip_0_696",
    "Azibo_ObsChimp_2018_07_11_c_clip_2000_3000",
    "Azibo_ObsChimp_2018_07_11_c_clip_8000_9000",
    "Azibo_ObsChimp_2018_07_11_c_clip_16000_17000",
    "Azibo_ObsChimp_2018_07_11_c_clip_19000_20000",
    "Azibo_ObsChimp_2018_08_06_a_clip_6000_7000",
    "Azibo_ObsChimp_2018_08_06_a_clip_8000_9000",
    "Azibo_ObsNatascha_2018_06_29_a_clip_14000_15000",
    "Azibo_ObsNatascha_2018_06_29_a_clip_17000_17712",
];

fn keyframe_block(image: &Json) -> Result<i64, BenchmarkError> {
    let file_name = image.get("file_name")?;
    let Json::Str(file_name) = file_name else {
        return Err(BenchmarkError::python(
            PyException::TypeError,
            format!(
                "argument should be a str or an os.PathLike object where __fspath__ returns \
                 a str, not '{}'",
                file_name.type_name()
            ),
        ));
    };
    let name = py_name(Path::new(file_name));
    let stem = py_stem(&name);
    match py_int(stem) {
        Ok(PyInt::Value(block)) => Ok(block),
        Ok(PyInt::OutOfRange { .. }) => Err(BenchmarkError::Unsupported(format!(
            "keyframe block {} is outside the 64-bit range",
            crate::pyfmt::repr_str(stem)
        ))),
        Err(err) => Err(BenchmarkError::python(
            PyException::ValueError,
            int_error_message(stem, err),
        )),
    }
}

fn lookup<'a, V>(map: &'a [(PyKey, V)], key: &PyKey) -> Option<&'a V> {
    map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn insert<V>(map: &mut Vec<(PyKey, V)>, key: PyKey, value: V) {
    match map.iter_mut().find(|(k, _)| *k == key) {
        Some(slot) => slot.1 = value,
        None => map.push((key, value)),
    }
}

struct RawTrack {
    frame: i64,
    track_id: Json,
    coords: [Coord; 4],
}

enum Coord {
    Raw(Json),
    Float(f64),
}

struct Blocks {
    by_block: BTreeMap<i64, Vec<Json>>,
    image_blocks: Vec<i64>,
}

fn keyframe_blocks(labels: &Json) -> Result<Blocks, BenchmarkError> {
    let mut image_id_to_block: Vec<(PyKey, i64)> = Vec::new();
    let mut image_blocks = Vec::new();
    for image in labels.get("images")?.iter()? {
        let id = image.get("id")?.clone();
        let block = keyframe_block(&image)?;
        image_blocks.push(block);
        insert(&mut image_id_to_block, id.key()?, block);
    }
    let mut by_block: BTreeMap<i64, Vec<Json>> = BTreeMap::new();
    for ann in labels.get("annotations")?.iter()? {
        if ann.get("bbox_id")?.eq_int(UNNAMED_BBOX_ID) {
            continue;
        }
        let image_id = ann.get("image_id")?;
        let block = *lookup(&image_id_to_block, &image_id.key()?)
            .ok_or_else(|| BenchmarkError::python(PyException::KeyError, image_id.repr()))?;
        by_block.entry(block).or_default().push(ann);
    }
    Ok(Blocks {
        by_block,
        image_blocks,
    })
}

fn interpolated_box(cur: &Json, next: &Json, t: f64) -> Result<[f64; 4], BenchmarkError> {
    let cur = cur.iter()?;
    let next = next.iter()?;
    let mut out = Vec::with_capacity(4);
    for i in 0.. {
        match (cur.get(i), next.get(i)) {
            (Some(c), Some(n)) => {
                let a = c.mul_float(1.0 - t)?;
                let b = n.mul_float(t)?;
                if out.len() == 4 {
                    return Err(BenchmarkError::python(
                        PyException::ValueError,
                        "too many values to unpack (expected 4)",
                    ));
                }
                out.push(a + b);
            }
            (Some(_), None) => {
                return Err(BenchmarkError::python(
                    PyException::ValueError,
                    "zip() argument 2 is shorter than argument 1",
                ));
            }
            (None, Some(_)) => {
                return Err(BenchmarkError::python(
                    PyException::ValueError,
                    "zip() argument 2 is longer than argument 1",
                ));
            }
            (None, None) => break,
        }
    }
    <[f64; 4]>::try_from(out).map_err(|out| {
        BenchmarkError::python(
            PyException::ValueError,
            format!(
                "not enough values to unpack (expected 4, got {})",
                out.len()
            ),
        )
    })
}

fn clip_tracks(labels: &Json) -> Result<(Vec<RawTrack>, Vec<i64>), BenchmarkError> {
    let blocks = keyframe_blocks(labels)?;
    let mut tracks = Vec::new();
    let overflow = |block: i64| {
        BenchmarkError::Unsupported(format!("keyframe block {block} is too large for a frame"))
    };
    for (&block, objs) in &blocks.by_block {
        let frame = block
            .checked_mul(KEYFRAME_STRIDE)
            .ok_or_else(|| overflow(block))?;
        for obj in objs {
            let [x, y, w, h] = obj.get("bbox")?.unpack4()?;
            tracks.push(RawTrack {
                frame,
                track_id: obj.get("bbox_id")?.clone(),
                coords: [Coord::Raw(x), Coord::Raw(y), Coord::Raw(w), Coord::Raw(h)],
            });
        }
        let mut next_by_id: Vec<(PyKey, &Json)> = Vec::new();
        let next_block = block.checked_add(1).ok_or_else(|| overflow(block))?;
        for o in blocks.by_block.get(&next_block).into_iter().flatten() {
            insert(&mut next_by_id, o.get("bbox_id")?.key()?, o);
        }
        for obj in objs {
            let matched = lookup(&next_by_id, &obj.get("bbox_id")?.key()?).copied();
            for offset in 1..KEYFRAME_STRIDE {
                let coords = match matched {
                    Some(next) => {
                        let t = offset as f64 / KEYFRAME_STRIDE as f64;
                        interpolated_box(obj.get("bbox")?, next.get("bbox")?, t)?.map(Coord::Float)
                    }
                    None => obj.get("bbox")?.unpack4()?.map(Coord::Raw),
                };
                tracks.push(RawTrack {
                    frame: frame.checked_add(offset).ok_or_else(|| overflow(block))?,
                    track_id: obj.get("bbox_id")?.clone(),
                    coords,
                });
            }
        }
    }
    Ok((tracks, blocks.image_blocks))
}

fn load_clip(label_path: &Path) -> Result<GtSequence, BenchmarkError> {
    let name = py_stem(&py_name(label_path)).to_owned();
    let labels = read_json(label_path)?;
    let (raw, image_blocks) = clip_tracks(&labels)?;
    if !labels.get("images")?.truthy() {
        return Err(BenchmarkError::EmptyGt(name));
    }
    let max_block = *image_blocks.iter().max().expect("images is not empty");
    let num_timesteps = (i128::from(max_block) + 1) * i128::from(KEYFRAME_STRIDE);
    let num_timesteps = usize::try_from(num_timesteps).map_err(|_| {
        BenchmarkError::Unsupported(format!(
            "{}: sequence length {num_timesteps} is negative or too large",
            py_str(label_path)
        ))
    })?;
    let coord = |c: &Coord, field: &str| match c {
        Coord::Raw(value) => json_f64(value, field, label_path),
        Coord::Float(f) => Ok(*f),
    };
    let tracks = raw
        .iter()
        .map(|t| {
            let [x, y, w, h] = &t.coords;
            Ok(Track {
                frame: t.frame,
                track_id: json_i64(&t.track_id, "bbox_id", label_path)?,
                x: coord(x, "bbox x")?,
                y: coord(y, "bbox y")?,
                w: coord(w, "bbox w")?,
                h: coord(h, "bbox h")?,
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

fn split_clip_names(label_dir: &Path, split: &str) -> Result<Vec<String>, BenchmarkError> {
    let mut names: Vec<String> = glob_suffix(label_dir, ".json")?
        .iter()
        .map(|name| py_stem(name).to_owned())
        .collect();
    names.sort();
    let in_val = |name: &String| VAL_CLIPS.contains(&name.as_str());
    let in_test = |name: &String| TEST_CLIPS.contains(&name.as_str());
    match split {
        "val" => names.retain(in_val),
        "test" => names.retain(in_test),
        "train" => names.retain(|name| !in_val(name) && !in_test(name)),
        _ => {
            return Err(BenchmarkError::UnknownSplit {
                benchmark: "chimpact",
                split: split.to_owned(),
                expected: "'train', 'val', or 'test'",
            });
        }
    }
    Ok(names)
}

pub fn load_chimpact(root: &Path, split: &str) -> Result<MOTDataset, BenchmarkError> {
    let label_dir = py_join(root, "ChimpACT_release_v1/labels");
    if !label_dir.is_dir() {
        return Err(BenchmarkError::LabelsDirNotFound(py_str(&label_dir)));
    }
    let sequences = split_clip_names(&label_dir, split)?
        .iter()
        .map(|name| load_clip(&py_join(&label_dir, format!("{name}.json"))).map(Sequence::Boxes))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MOTDataset {
        name: "chimpact".to_owned(),
        split: split.to_owned(),
        sequences,
        protocol: Protocol::new("chimpact", FrameConvention::new("0-indexed", 0), vec![1]),
    })
}
