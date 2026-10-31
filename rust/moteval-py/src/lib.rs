//! PyO3 module behind the `moteval` Python package: `moteval._moteval`.
//!
//! The boundary is plain data. Python keeps its frozen dataclasses (`MOTDataset`,
//! `GtSequence`, `Track`, `Protocol`, ...), so every public name in `moteval/__init__.py`
//! keeps its type and `dataclasses.replace` keeps working; this module reads them by
//! attribute into Rust-owned values, runs the Rust code with the GIL released, and hands
//! results back as dicts of `float` and numpy arrays (Rust vectors moved into numpy, no
//! copy). Rust errors become `ValueError` with the Rust message, which equals Python's;
//! an I/O error becomes `OSError`, as in Python.
//!
//! - `evaluate(dataset, predictions, metrics)`: `moteval.eval.evaluate`.
//! - `results_table`, `results_to_json`, `results_to_csv`: the CLI table and the JSON and
//!   CSV exports of a Python `EvaluationResult`.
//! - `cli_run(argv)`: the CLI with captured output; `main()`: the CLI on `sys.argv`, the
//!   entry point for the `moteval` console script.

use std::path::PathBuf;

use moteval::cli;
use moteval::eval::{self, EvalError};
use moteval::formats::{MaskTrack, Track};
use moteval::metrics::{Clear, Count, Hota, Identity, JAndF, Metric, Scores, TrackMap, Value};
use moteval::model::{
    FrameConvention, GtSequence, MOTDataset, MaskGtSequence, Matrix, Protocol, Sequence,
};
use moteval::results::{self, ClassResult, EvaluationResult, MetricScores};
use numpy::prelude::*;
use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

// ------------------------------------------------------------------ Python -> Rust

/// The Rust metric for a Python metric object (by class name, as `evaluate` keys
/// results) or a metric class name.
fn metric_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn Metric>> {
    let name: String = if let Ok(name) = obj.cast::<PyString>() {
        name.to_str()?.to_string()
    } else {
        obj.get_type().name()?.to_str()?.to_string()
    };
    Ok(match name.as_str() {
        "HOTA" => Box::new(Hota),
        "CLEAR" => Box::new(Clear),
        "Identity" => Box::new(Identity),
        "Count" => Box::new(Count),
        "JAndF" => Box::new(JAndF),
        "TrackMAP" => Box::new(TrackMap),
        other => {
            return Err(PyValueError::new_err(format!(
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

fn rows<T>(
    seq: &Bound<'_, PyAny>,
    attr: &str,
    convert: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    seq.getattr(attr)?
        .try_iter()?
        .map(|row| convert(&row?))
        .collect()
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
/// a `moteval.data.model.MaskGtSequence`, as `evaluate` decides in Python.
fn dataset_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<MOTDataset> {
    let mask_class = py.import("moteval.data.model")?.getattr("MaskGtSequence")?;
    let mut sequences = Vec::new();
    for seq in obj.getattr("sequences")?.try_iter()? {
        let seq = seq?;
        let name: String = seq.getattr("name")?.extract()?;
        let num_timesteps: usize = seq.getattr("num_timesteps")?.extract()?;
        sequences.push(if seq.is_instance(&mask_class)? {
            Sequence::Masks(MaskGtSequence {
                name,
                num_timesteps,
                tracks: rows(&seq, "tracks", mask_track_from_py)?,
                ignore_regions: rows(&seq, "ignore_regions", mask_track_from_py)?,
            })
        } else {
            Sequence::Boxes(GtSequence {
                name,
                num_timesteps,
                tracks: rows(&seq, "tracks", track_from_py)?,
                ignore_regions: rows(&seq, "ignore_regions", track_from_py)?,
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

/// A score value: a number, or a numpy array of up to two dimensions (read as float64,
/// C order).
fn value_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let numpy = py.import("numpy")?;
    if !obj.is_instance(&numpy.getattr("ndarray")?)? {
        return Ok(Value::Float(obj.extract()?));
    }
    let contiguous = numpy.call_method1("ascontiguousarray", (obj, numpy.getattr("float64")?))?;
    let array: PyReadonlyArrayDyn<'_, f64> = contiguous.extract()?;
    let data = array
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .to_vec();
    match array.shape() {
        [] => Ok(Value::Float(data[0])),
        [_] => Ok(Value::Array(data)),
        [rows, cols] => Matrix::new(*rows, *cols, data)
            .map(Value::Matrix)
            .map_err(|err| PyValueError::new_err(err.to_string())),
        shape => Err(PyValueError::new_err(format!(
            "a score array has {} dimensions; at most 2 are supported",
            shape.len()
        ))),
    }
}

/// `{metric: {field: value}}`.
fn metric_scores_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<MetricScores> {
    let mut out = MetricScores::new();
    for (metric, fields) in obj.cast::<PyDict>()?.iter() {
        let mut scores = Scores::new();
        for (field, value) in fields.cast::<PyDict>()?.iter() {
            scores.insert(field.extract::<String>()?, value_from_py(py, &value)?);
        }
        out.insert(metric.extract::<String>()?, scores);
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

/// `{metric: {field: value}}`. Values are cloned out of the shared `Scores` map (a
/// handful of floats each); arrays then move into numpy without another copy.
fn metric_scores_to_py<'py>(
    py: Python<'py>,
    scores: &MetricScores,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (metric, fields) in scores.iter() {
        let inner = PyDict::new(py);
        for (field, value) in fields.iter() {
            inner.set_item(field, value_to_py(py, value.clone())?)?;
        }
        out.set_item(metric, inner)?;
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

fn eval_error_to_py(err: EvalError) -> PyErr {
    let message = err.to_string();
    match err.python_exception() {
        "OSError" => PyOSError::new_err(message),
        "TypeError" => PyTypeError::new_err(message),
        "IndexError" => PyIndexError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

// ------------------------------------------------------------------------ functions

/// `evaluate(dataset, predictions, metrics)`: score a Python `MOTDataset` against the
/// `<seq>.txt` files in `predictions`. `metrics` holds metric objects or class names.
/// Returns the keyword arguments of `EvaluationResult` (`per_class` values as those of
/// `ClassResult`). The GIL is released while Rust reads and scores.
#[pyfunction]
fn evaluate<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    predictions: PathBuf,
    metrics: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let dataset = dataset_from_py(py, dataset)?;
    let metrics: Vec<Box<dyn Metric>> = metrics
        .try_iter()?
        .map(|metric| metric_from_py(&metric?))
        .collect::<PyResult<_>>()?;
    let result = py
        .detach(|| eval::evaluate(&dataset, &predictions, &metrics))
        .map_err(eval_error_to_py)?;
    result_to_py(py, &result)
}

/// The headline table `moteval run` prints for a Python `EvaluationResult`, without the
/// final newline (`moteval.cli._table`).
#[pyfunction]
fn results_table(py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<String> {
    let result = result_from_py(py, result)?;
    cli::table(&result).map_err(|err| PyValueError::new_err(err.to_string()))
}

/// `json.dumps(to_json_dict(result, dataset=dataset, split=split))`.
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
    module.add_function(wrap_pyfunction!(evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(results_table, module)?)?;
    module.add_function(wrap_pyfunction!(results_to_json, module)?)?;
    module.add_function(wrap_pyfunction!(results_to_csv, module)?)?;
    module.add_function(wrap_pyfunction!(cli_run, module)?)?;
    module.add_function(wrap_pyfunction!(main, module)?)?;
    Ok(())
}
