//! PyO3 module behind the `moteval` Python package: `moteval._moteval`.
//!
//! The Python package keeps only its public data types (frozen dataclasses such as
//! `MOTDataset`, `GtSequence`, `Track`, `Protocol`, `SequenceData`) and thin functions
//! that call this module. The boundary is plain data: this module reads the dataclasses by
//! attribute into Rust-owned values and returns tuples, lists, `float`s, and numpy arrays
//! (Rust vectors moved into numpy without a copy), from which Python builds its
//! dataclasses. Long Rust work (loading, reading, evaluating) runs with the GIL released.
//!
//! Every Rust error is raised as `RustError(exception, message)`: `exception` names the
//! Python class the pure-Python code raised (`"ValueError"`, `"OSError"`, `"KeyError"`,
//! ...) and `message` is its text. `moteval._rust.call` re-raises it as that class.
//!
//! - Loaders: `load_benchmark(name, root, split)`, `load_motchallenge`, `load_mots`,
//!   `default_data_root()`, and `benchmarks_table()`.
//! - Formats: `read_mot`, `read_mots`, `write_mot`, `write_mots`.
//! - Metrics: `metric_info(name)`, `metric_eval_sequence(name, data)`, and
//!   `metric_combine(name, method, all_res)`.
//! - `evaluate(dataset, predictions, metric_names)`: `moteval.evaluate`.
//! - `results_table`, `results_to_json`, `results_to_csv`: the CLI table and the JSON and
//!   CSV exports of a Python `EvaluationResult`.
//! - `cli_run(argv)`: the CLI with captured output; `main()`: the CLI on `sys.argv`, the
//!   entry point for the `moteval` console script.

use std::path::PathBuf;

use moteval::benchmarks::{self, BENCHMARKS, BenchmarkError};
use moteval::cli;
use moteval::eval::{self, EvalError};
use moteval::formats::{self, FormatError, MaskTrack, Track};
use moteval::metrics::{
    Clear, Count, Hota, Identity, JAndF, Metric, MetricError, Scores, TrackMap, Value,
};
use moteval::model::{
    FrameConvention, Geometry, GtSequence, MOTDataset, MaskGtSequence, Matrix, ModelError,
    Protocol, Sequence, SequenceData, SequenceDataParts,
};
use moteval::results::{self, ClassResult, EvaluationResult, MetricScores};
use moteval::rle::RleMask;
use numpy::prelude::*;
use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString};

create_exception!(
    _moteval,
    RustError,
    PyException,
    "A Rust error: args are (Python exception class name, message)."
);

/// `RustError(exception, message)`.
fn rust_error(exception: &str, message: String) -> PyErr {
    RustError::new_err((exception.to_string(), message))
}

fn format_error(err: FormatError) -> PyErr {
    rust_error(err.python_exception(), err.to_string())
}

fn benchmark_error(err: BenchmarkError) -> PyErr {
    rust_error(err.python_exception(), err.to_string())
}

fn eval_error(err: EvalError) -> PyErr {
    rust_error(err.python_exception(), err.to_string())
}

fn metric_error(err: MetricError) -> PyErr {
    rust_error(err.python_exception(), err.to_string())
}

fn value_error(message: impl Into<String>) -> PyErr {
    rust_error("ValueError", message.into())
}

// ------------------------------------------------------------------ Python -> Rust

/// The Rust metric of a built-in metric class name (`type(metric).__name__`).
fn metric_by_name(name: &str) -> PyResult<Box<dyn Metric>> {
    Ok(match name {
        "HOTA" => Box::new(Hota),
        "CLEAR" => Box::new(Clear),
        "Identity" => Box::new(Identity),
        "Count" => Box::new(Count),
        "JAndF" => Box::new(JAndF),
        "TrackMAP" => Box::new(TrackMap),
        other => {
            return Err(value_error(format!(
                "metric {other} has no Rust implementation"
            )));
        }
    })
}

fn track_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Track> {
    Ok(Track {
        frame: obj.getattr("frame")?.extract()?,
        track_id: obj.getattr("track_id")?.extract()?,
        x: obj.getattr("x")?.extract()?,
        y: obj.getattr("y")?.extract()?,
        w: obj.getattr("w")?.extract()?,
        h: obj.getattr("h")?.extract()?,
        conf: obj.getattr("conf")?.extract()?,
        class_id: obj.getattr("class_id")?.extract()?,
    })
}

fn mask_track_from_py(obj: &Bound<'_, PyAny>) -> PyResult<MaskTrack> {
    Ok(MaskTrack {
        frame: obj.getattr("frame")?.extract()?,
        track_id: obj.getattr("track_id")?.extract()?,
        class_id: obj.getattr("class_id")?.extract()?,
        img_h: obj.getattr("img_h")?.extract()?,
        img_w: obj.getattr("img_w")?.extract()?,
        rle: obj.getattr("rle")?.extract()?,
    })
}

fn rows_of<T>(
    rows: &Bound<'_, PyAny>,
    convert: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    rows.try_iter()?.map(|row| convert(&row?)).collect()
}

fn protocol_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Protocol> {
    let convention = obj.getattr("frame_convention")?;
    Ok(Protocol {
        name: obj.getattr("name")?.extract()?,
        frame_convention: FrameConvention::new(
            convention.getattr("name")?.extract::<String>()?,
            convention.getattr("first_frame")?.extract()?,
        ),
        eval_classes: obj.getattr("eval_classes")?.extract()?,
        distractor_classes: obj.getattr("distractor_classes")?.extract()?,
        distractor_iou_threshold: obj.getattr("distractor_iou_threshold")?.extract()?,
        ignore_iou_threshold: obj.getattr("ignore_iou_threshold")?.extract()?,
        drop_zero_conf_gt: obj.getattr("drop_zero_conf_gt")?.extract()?,
        matching_fill: obj.getattr("matching_fill")?.extract()?,
    })
}

/// A Python `MOTDataset` as a Rust one. A sequence is a mask sequence exactly when it is
/// a `moteval.data.model.MaskGtSequence`, as Python's `evaluate` decided.
fn dataset_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<MOTDataset> {
    let mask_class = py.import("moteval.data.model")?.getattr("MaskGtSequence")?;
    let mut sequences = Vec::new();
    for seq in obj.getattr("sequences")?.try_iter()? {
        let seq = seq?;
        let name: String = seq.getattr("name")?.extract()?;
        let num_timesteps: usize = seq.getattr("num_timesteps")?.extract()?;
        let tracks = seq.getattr("tracks")?;
        let ignore = seq.getattr("ignore_regions")?;
        sequences.push(if seq.is_instance(&mask_class)? {
            Sequence::Masks(MaskGtSequence {
                name,
                num_timesteps,
                tracks: rows_of(&tracks, mask_track_from_py)?,
                ignore_regions: rows_of(&ignore, mask_track_from_py)?,
            })
        } else {
            Sequence::Boxes(GtSequence {
                name,
                num_timesteps,
                tracks: rows_of(&tracks, track_from_py)?,
                ignore_regions: rows_of(&ignore, track_from_py)?,
            })
        });
    }
    Ok(MOTDataset {
        name: obj.getattr("name")?.extract()?,
        split: obj.getattr("split")?.extract()?,
        sequences,
        protocol: protocol_from_py(&obj.getattr("protocol")?)?,
    })
}

/// `np.ascontiguousarray(obj, dtype)` as a flat vector plus its shape.
fn flat_array<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<(Vec<T>, Vec<usize>)> {
    let numpy = py.import("numpy")?;
    let contiguous = numpy.call_method1("ascontiguousarray", (obj, numpy.getattr(dtype)?))?;
    let array: PyReadonlyArrayDyn<'py, T> = contiguous.extract()?;
    let data = array
        .as_slice()
        .map_err(|err| value_error(err.to_string()))?
        .to_vec();
    Ok((data, array.shape().to_vec()))
}

/// A tuple of per-frame 1-D arrays.
fn frames_of<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Vec<Vec<T>>> {
    obj.try_iter()?
        .map(|frame| Ok(flat_array::<T>(py, &frame?, dtype)?.0))
        .collect()
}

/// Per-frame ids as indices; a negative id cannot be dense.
fn id_frames(py: Python<'_>, obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<Vec<usize>>> {
    frames_of::<i64>(py, obj, "int64")?
        .into_iter()
        .map(|ids| {
            ids.into_iter()
                .map(|id| {
                    usize::try_from(id).map_err(|_| value_error(format!("negative {what} {id}")))
                })
                .collect()
        })
        .collect()
}

/// A tuple of per-frame `(n, 4)` box arrays.
fn box_frames(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<[f64; 4]>>> {
    obj.try_iter()?
        .map(|frame| {
            let (data, _) = flat_array::<f64>(py, &frame?, "float64")?;
            let (boxes, rest) = data.as_chunks::<4>();
            if !rest.is_empty() {
                return Err(value_error("a box array does not hold 4 columns"));
            }
            Ok(boxes.to_vec())
        })
        .collect()
}

/// A tuple of per-frame object arrays of pycocotools RLE dicts
/// (`{"size": [h, w], "counts": bytes | str}`).
fn mask_frames(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<RleMask>>> {
    obj.try_iter()?
        .map(|frame| {
            frame?
                .try_iter()?
                .map(|rle| {
                    let rle = rle?;
                    // pycocotools gives `size` as a list; accept a tuple as well.
                    let size: Vec<usize> = rle.get_item("size")?.extract()?;
                    let [h, w] = size[..] else {
                        return Err(value_error("an RLE size must hold 2 values"));
                    };
                    let counts = rle.get_item("counts")?;
                    let bytes: Vec<u8> = match counts.cast::<PyString>() {
                        Ok(text) => text.to_str()?.as_bytes().to_vec(),
                        Err(_) => counts.cast::<PyBytes>()?.as_bytes().to_vec(),
                    };
                    RleMask::from_compressed(h, w, &bytes)
                        .map_err(|err| value_error(err.to_string()))
                })
                .collect()
        })
        .collect()
}

/// A Python `SequenceData` as a Rust one (checked by `SequenceData::new`).
fn sequence_data_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<SequenceData> {
    let geometry = obj.getattr("geometry")?;
    let model = py.import("moteval.data.model")?;
    let geometry = if geometry.is_instance(&model.getattr("MaskGeometry")?)? {
        Geometry::Masks {
            gt: mask_frames(&geometry.getattr("gt")?)?,
            pred: mask_frames(&geometry.getattr("pred")?)?,
        }
    } else {
        Geometry::Boxes {
            gt: box_frames(py, &geometry.getattr("gt")?)?,
            pred: box_frames(py, &geometry.getattr("pred")?)?,
        }
    };
    let similarity = obj
        .getattr("similarity")?
        .try_iter()?
        .map(|frame| {
            let (data, shape) = flat_array::<f64>(py, &frame?, "float64")?;
            match shape[..] {
                [rows, cols] => Matrix::new(rows, cols, data).map_err(model_error),
                _ => Err(value_error("a similarity matrix is not 2-D")),
            }
        })
        .collect::<PyResult<Vec<Matrix>>>()?;
    SequenceData::new(SequenceDataParts {
        name: obj.getattr("name")?.extract()?,
        num_timesteps: obj.getattr("num_timesteps")?.extract()?,
        num_gt_ids: obj.getattr("num_gt_ids")?.extract()?,
        num_pred_ids: obj.getattr("num_pred_ids")?.extract()?,
        num_gt_dets: obj.getattr("num_gt_dets")?.extract()?,
        num_pred_dets: obj.getattr("num_pred_dets")?.extract()?,
        gt_ids: id_frames(py, &obj.getattr("gt_ids")?, "gt id")?,
        pred_ids: id_frames(py, &obj.getattr("pred_ids")?, "pred id")?,
        pred_confidences: frames_of::<f64>(py, &obj.getattr("pred_confidences")?, "float64")?,
        geometry,
        similarity,
    })
    .map_err(model_error)
}

fn model_error(err: ModelError) -> PyErr {
    metric_error(MetricError::Model(err))
}

/// A score value: a number, or a numpy array of up to two dimensions (read as float64,
/// C order).
fn value_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let numpy = py.import("numpy")?;
    if !obj.is_instance(&numpy.getattr("ndarray")?)? {
        return Ok(Value::Float(obj.extract()?));
    }
    let (data, shape) = flat_array::<f64>(py, obj, "float64")?;
    match shape[..] {
        [] => Ok(Value::Float(data[0])),
        [_] => Ok(Value::Array(data)),
        [rows, cols] => Matrix::new(rows, cols, data)
            .map(Value::Matrix)
            .map_err(model_error),
        _ => Err(value_error(format!(
            "a score array has {} dimensions; at most 2 are supported",
            shape.len()
        ))),
    }
}

/// `{field: value}`.
fn scores_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Scores> {
    let mut scores = Scores::new();
    for (field, value) in obj.cast::<PyDict>()?.iter() {
        scores.insert(field.extract::<String>()?, value_from_py(py, &value)?);
    }
    Ok(scores)
}

/// `{metric: {field: value}}`.
fn metric_scores_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<MetricScores> {
    let mut out = MetricScores::new();
    for (metric, fields) in obj.cast::<PyDict>()?.iter() {
        out.insert(metric.extract::<String>()?, scores_from_py(py, &fields)?);
    }
    Ok(out)
}

fn per_sequence_from_py(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Vec<(String, MetricScores)>> {
    obj.cast::<PyDict>()?
        .iter()
        .map(|(seq, scores)| Ok((seq.extract()?, metric_scores_from_py(py, &scores)?)))
        .collect()
}

/// A Python `EvaluationResult` as a Rust one, mappings in their dict order.
fn result_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<EvaluationResult> {
    let mut per_class = Vec::new();
    for (cls_id, class_result) in obj.getattr("per_class")?.cast::<PyDict>()?.iter() {
        per_class.push((
            cls_id.extract()?,
            ClassResult {
                per_sequence: per_sequence_from_py(py, &class_result.getattr("per_sequence")?)?,
                combined: metric_scores_from_py(py, &class_result.getattr("combined")?)?,
            },
        ));
    }
    Ok(EvaluationResult {
        per_sequence: per_sequence_from_py(py, &obj.getattr("per_sequence")?)?,
        combined: metric_scores_from_py(py, &obj.getattr("combined")?)?,
        per_class,
        class_averaged: metric_scores_from_py(py, &obj.getattr("class_averaged")?)?,
        det_averaged: metric_scores_from_py(py, &obj.getattr("det_averaged")?)?,
    })
}

// ------------------------------------------------------------------ Rust -> Python

/// `(frame, track_id, x, y, w, h, conf, class_id)`: the `Track` fields in order.
type TrackRow = (i64, i64, f64, f64, f64, f64, f64, i64);
/// `(frame, track_id, class_id, img_h, img_w, rle)`: the `MaskTrack` fields in order.
type MaskTrackRow = (i64, i64, i64, i64, i64, String);

fn track_row(t: &Track) -> TrackRow {
    (t.frame, t.track_id, t.x, t.y, t.w, t.h, t.conf, t.class_id)
}

fn mask_track_row(t: &MaskTrack) -> MaskTrackRow {
    (
        t.frame,
        t.track_id,
        t.class_id,
        t.img_h,
        t.img_w,
        t.rle.clone(),
    )
}

/// `(name, (convention name, first_frame), eval_classes, distractor_classes,
/// distractor_iou_threshold, ignore_iou_threshold, drop_zero_conf_gt, matching_fill)`.
type ProtocolRow = (
    String,
    (String, i64),
    Vec<i64>,
    Vec<i64>,
    f64,
    f64,
    bool,
    f64,
);

fn protocol_row(p: &Protocol) -> ProtocolRow {
    (
        p.name.clone(),
        (
            p.frame_convention.name.clone(),
            p.frame_convention.first_frame,
        ),
        p.eval_classes.clone(),
        p.distractor_classes.clone(),
        p.distractor_iou_threshold,
        p.ignore_iou_threshold,
        p.drop_zero_conf_gt,
        p.matching_fill,
    )
}

/// A loaded dataset as `(name, split, protocol, sequences)`; each sequence is
/// `("boxes" | "masks", name, num_timesteps, tracks, ignore_regions)` with rows as tuples.
fn dataset_to_py<'py>(py: Python<'py>, dataset: &MOTDataset) -> PyResult<Bound<'py, PyAny>> {
    let mut sequences = Vec::with_capacity(dataset.sequences.len());
    for seq in &dataset.sequences {
        let row = match seq {
            Sequence::Boxes(s) => (
                "boxes",
                s.name.as_str(),
                s.num_timesteps,
                s.tracks.iter().map(track_row).collect::<Vec<_>>(),
                s.ignore_regions.iter().map(track_row).collect::<Vec<_>>(),
            )
                .into_pyobject(py)?
                .into_any(),
            Sequence::Masks(s) => (
                "masks",
                s.name.as_str(),
                s.num_timesteps,
                s.tracks.iter().map(mask_track_row).collect::<Vec<_>>(),
                s.ignore_regions
                    .iter()
                    .map(mask_track_row)
                    .collect::<Vec<_>>(),
            )
                .into_pyobject(py)?
                .into_any(),
        };
        sequences.push(row);
    }
    Ok((
        dataset.name.as_str(),
        dataset.split.as_str(),
        protocol_row(&dataset.protocol),
        sequences,
    )
        .into_pyobject(py)?
        .into_any())
}

/// A value as Python moteval holds it: a `float`, or a float64 numpy array that takes
/// over the Rust vector without a copy.
fn value_to_py<'py>(py: Python<'py>, value: Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Float(x) => x.into_pyobject(py)?.into_any(),
        Value::Array(values) => PyArray1::from_vec(py, values).into_any(),
        Value::Matrix(matrix) => {
            let (rows, cols) = matrix.shape();
            PyArray1::from_vec(py, matrix.into_data())
                .reshape([rows, cols])?
                .into_any()
        }
    })
}

/// `{field: value}`, in the metric's insertion order.
fn scores_to_py<'py>(py: Python<'py>, scores: &Scores) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (field, value) in scores.iter() {
        out.set_item(field, value_to_py(py, value.clone())?)?;
    }
    Ok(out)
}

/// `{metric: {field: value}}`. Values are cloned out of the shared `Scores` map (a
/// handful of floats each); arrays then move into numpy without another copy.
fn metric_scores_to_py<'py>(
    py: Python<'py>,
    scores: &MetricScores,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (metric, fields) in scores.iter() {
        out.set_item(metric, scores_to_py(py, fields)?)?;
    }
    Ok(out)
}

fn per_sequence_to_py<'py>(
    py: Python<'py>,
    per_sequence: &[(String, MetricScores)],
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (seq, scores) in per_sequence {
        out.set_item(seq, metric_scores_to_py(py, scores)?)?;
    }
    Ok(out)
}

/// The result as the keyword arguments of `EvaluationResult`, with each `per_class`
/// value as the keyword arguments of `ClassResult`.
fn result_to_py<'py>(py: Python<'py>, result: &EvaluationResult) -> PyResult<Bound<'py, PyDict>> {
    let per_class = PyDict::new(py);
    for (cls_id, class_result) in &result.per_class {
        let entry = PyDict::new(py);
        entry.set_item(
            "per_sequence",
            per_sequence_to_py(py, &class_result.per_sequence)?,
        )?;
        entry.set_item("combined", metric_scores_to_py(py, &class_result.combined)?)?;
        per_class.set_item(cls_id, entry)?;
    }
    let out = PyDict::new(py);
    out.set_item(
        "per_sequence",
        per_sequence_to_py(py, &result.per_sequence)?,
    )?;
    out.set_item("combined", metric_scores_to_py(py, &result.combined)?)?;
    out.set_item("per_class", per_class)?;
    out.set_item(
        "class_averaged",
        metric_scores_to_py(py, &result.class_averaged)?,
    )?;
    out.set_item(
        "det_averaged",
        metric_scores_to_py(py, &result.det_averaged)?,
    )?;
    Ok(out)
}

// ------------------------------------------------------------------------- loaders

/// The built-in benchmarks as `(name, default split)`, in `BENCHMARKS` order.
#[pyfunction]
fn benchmarks_table() -> Vec<(&'static str, &'static str)> {
    BENCHMARKS
        .iter()
        .map(|(name, _, split)| (*name, *split))
        .collect()
}

/// `default_data_root()`: `MOTEVAL_DATA_ROOT` when set (an empty value is an error), else
/// `data/benchmarks`.
#[pyfunction]
fn default_data_root() -> PyResult<PathBuf> {
    benchmarks::default_data_root().map_err(benchmark_error)
}

/// `load_dataset(name, root, split)`: a built-in benchmark by name. `root` defaults to
/// `default_data_root() / name` and `split` to the benchmark's default split.
#[pyfunction]
#[pyo3(signature = (name, root=None, split=None))]
fn load_benchmark<'py>(
    py: Python<'py>,
    name: &str,
    root: Option<PathBuf>,
    split: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let dataset = py
        .detach(|| benchmarks::load_dataset(name, root.as_deref(), split.as_deref()))
        .map_err(benchmark_error)?;
    dataset_to_py(py, &dataset)
}

/// `load_motchallenge(root, split)`: any directory in the MOTChallenge box layout.
#[pyfunction]
fn load_motchallenge<'py>(
    py: Python<'py>,
    root: PathBuf,
    split: String,
) -> PyResult<Bound<'py, PyAny>> {
    let dataset = py
        .detach(|| benchmarks::motchallenge::load_motchallenge(&root, &split))
        .map_err(benchmark_error)?;
    dataset_to_py(py, &dataset)
}

/// `load_mots(root, split, class_id)`: any directory in the MOTS20 mask layout.
#[pyfunction]
fn load_mots<'py>(
    py: Python<'py>,
    root: PathBuf,
    split: String,
    class_id: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let dataset = py
        .detach(|| benchmarks::mots20::load_mots(&root, &split, class_id))
        .map_err(benchmark_error)?;
    dataset_to_py(py, &dataset)
}

// ------------------------------------------------------------------------- formats

/// `read_mot(path, class_column=...)` as `Track` field tuples.
#[pyfunction]
fn read_mot(py: Python<'_>, path: PathBuf, class_column: bool) -> PyResult<Vec<TrackRow>> {
    let tracks = py
        .detach(|| formats::read_mot(&path, class_column))
        .map_err(format_error)?;
    Ok(tracks.iter().map(track_row).collect())
}

/// `read_mots(path)` as `MaskTrack` field tuples.
#[pyfunction]
fn read_mots(py: Python<'_>, path: PathBuf) -> PyResult<Vec<MaskTrackRow>> {
    let tracks = py
        .detach(|| formats::read_mots(&path))
        .map_err(format_error)?;
    Ok(tracks.iter().map(mask_track_row).collect())
}

/// `write_mot(path, tracks)`: `Track` objects as MOTChallenge rows.
#[pyfunction]
fn write_mot(py: Python<'_>, path: PathBuf, tracks: &Bound<'_, PyAny>) -> PyResult<()> {
    let tracks = rows_of(tracks, track_from_py)?;
    py.detach(|| formats::write_mot(&path, &tracks))
        .map_err(format_error)
}

/// `write_mots(path, tracks)`: `MaskTrack` objects as MOTS rows.
#[pyfunction]
fn write_mots(py: Python<'_>, path: PathBuf, tracks: &Bound<'_, PyAny>) -> PyResult<()> {
    let tracks = rows_of(tracks, mask_track_from_py)?;
    py.detach(|| formats::write_mots(&path, &tracks))
        .map_err(format_error)
}

// ------------------------------------------------------------------------- metrics

/// `(fields, geometry)` of a built-in metric: its public result fields in order and the
/// geometry it reads (`"boxes"`, `"masks"`, or `None`).
#[pyfunction]
fn metric_info(name: &str) -> PyResult<(Vec<&'static str>, Option<&'static str>)> {
    let metric = metric_by_name(name)?;
    Ok((
        metric.fields().to_vec(),
        metric.geometry().map(|kind| kind.as_str()),
    ))
}

/// `metric.eval_sequence(data)` for a built-in metric, on a Python `SequenceData`.
#[pyfunction]
fn metric_eval_sequence<'py>(
    py: Python<'py>,
    name: &str,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let metric = metric_by_name(name)?;
    let data = sequence_data_from_py(py, data)?;
    let scores = py
        .detach(|| metric.eval_sequence(&data))
        .map_err(metric_error)?;
    scores_to_py(py, &scores)
}

/// `metric.<method>(all_res)` for a built-in metric: `method` is `combine_sequences`,
/// `combine_classes_class_averaged`, or `combine_classes_det_averaged`, and `all_res`
/// maps a key to its `Scores` dict (combined in dict order).
#[pyfunction]
fn metric_combine<'py>(
    py: Python<'py>,
    name: &str,
    method: &str,
    all_res: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let metric = metric_by_name(name)?;
    let results: Vec<Scores> = all_res
        .cast::<PyDict>()?
        .values()
        .iter()
        .map(|scores| scores_from_py(py, &scores))
        .collect::<PyResult<_>>()?;
    let refs: Vec<&Scores> = results.iter().collect();
    let combined = match method {
        "combine_sequences" => metric.combine_sequences(&refs),
        "combine_classes_class_averaged" => metric.combine_classes_class_averaged(&refs),
        "combine_classes_det_averaged" => metric.combine_classes_det_averaged(&refs),
        other => return Err(value_error(format!("unknown combiner {other}"))),
    }
    .map_err(metric_error)?;
    scores_to_py(py, &combined)
}

// ---------------------------------------------------------------------- evaluation

/// `evaluate(dataset, predictions, metric_names)`: score a Python `MOTDataset` against
/// the `<seq>.txt` files in `predictions` with the named built-in metrics. Returns the
/// keyword arguments of `EvaluationResult` (`per_class` values as those of
/// `ClassResult`). The GIL is released while Rust reads and scores.
#[pyfunction]
fn evaluate<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    predictions: PathBuf,
    metric_names: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let dataset = dataset_from_py(py, dataset)?;
    let metrics: Vec<Box<dyn Metric>> = metric_names
        .iter()
        .map(|name| metric_by_name(name))
        .collect::<PyResult<_>>()?;
    let result = py
        .detach(|| eval::evaluate(&dataset, &predictions, &metrics))
        .map_err(eval_error)?;
    result_to_py(py, &result)
}

/// The headline table `moteval run` prints for a Python `EvaluationResult`, without the
/// final newline.
#[pyfunction]
fn results_table(py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<String> {
    let result = result_from_py(py, result)?;
    cli::table(&result).map_err(|err| value_error(err.to_string()))
}

/// The JSON text `moteval run --out-json` writes (`json.dumps` of the result's export).
#[pyfunction]
fn results_to_json(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    dataset: &str,
    split: &str,
) -> PyResult<String> {
    Ok(results::to_json(
        &result_from_py(py, result)?,
        dataset,
        split,
    ))
}

/// The long-form CSV text `moteval run --out-csv` writes.
#[pyfunction]
fn results_to_csv(py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(results::to_csv(&result_from_py(py, result)?))
}

/// Run the CLI on `argv` (without the program name) with captured output. Returns
/// `(exit_code, stdout, stderr)`. The GIL is released while it runs.
#[pyfunction]
fn cli_run(py: Python<'_>, argv: Vec<String>) -> (i32, String, String) {
    let (code, stdout, stderr) = py.detach(|| {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = cli::run(&argv, &mut stdout, &mut stderr);
        (code, stdout, stderr)
    });
    (
        code,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

/// The `moteval` console script: run the CLI on `sys.argv[1:]`, writing to the process's
/// stdout and stderr. Returns the exit code.
#[pyfunction]
fn main(py: Python<'_>) -> PyResult<i32> {
    let argv: Vec<String> = py.import("sys")?.getattr("argv")?.extract()?;
    let args = argv.get(1..).unwrap_or_default().to_vec();
    Ok(py.detach(|| cli::main(&args)))
}

#[pymodule]
fn _moteval(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RustError", module.py().get_type::<RustError>())?;
    module.add_function(wrap_pyfunction!(benchmarks_table, module)?)?;
    module.add_function(wrap_pyfunction!(default_data_root, module)?)?;
    module.add_function(wrap_pyfunction!(load_benchmark, module)?)?;
    module.add_function(wrap_pyfunction!(load_motchallenge, module)?)?;
    module.add_function(wrap_pyfunction!(load_mots, module)?)?;
    module.add_function(wrap_pyfunction!(read_mot, module)?)?;
    module.add_function(wrap_pyfunction!(read_mots, module)?)?;
    module.add_function(wrap_pyfunction!(write_mot, module)?)?;
    module.add_function(wrap_pyfunction!(write_mots, module)?)?;
    module.add_function(wrap_pyfunction!(metric_info, module)?)?;
    module.add_function(wrap_pyfunction!(metric_eval_sequence, module)?)?;
    module.add_function(wrap_pyfunction!(metric_combine, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(results_table, module)?)?;
    module.add_function(wrap_pyfunction!(results_to_json, module)?)?;
    module.add_function(wrap_pyfunction!(results_to_csv, module)?)?;
    module.add_function(wrap_pyfunction!(cli_run, module)?)?;
    module.add_function(wrap_pyfunction!(main, module)?)?;
    Ok(())
}
