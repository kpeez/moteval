use std::fmt;
use std::path::{Path, PathBuf};

use crate::convert::{ConvertError, build_mask_sequence_data, build_sequence_data};
use crate::formats::pytext::py_join;
use crate::formats::{
    FormatError, MaskTrack, Track, missing_prediction_files, read_mot, read_mots,
};
use crate::metrics::{Metric, MetricError, Scores};
use crate::model::{MOTDataset, Sequence, SequenceData};
use crate::pyfmt;
use crate::results::{ClassResult, EvaluationResult, MetricScores};

#[derive(Debug)]
pub enum EvalError {
    DuplicateMetrics(Vec<String>),
    NoEvalClasses {
        protocol: String,
    },
    RepeatedClass {
        protocol: String,
        classes: Vec<i64>,
    },
    NoSequences {
        dataset: String,
    },
    GeometryMismatch {
        metric: String,
        reads: &'static str,
        sequence: String,
        dataset: String,
        holds: &'static str,
    },
    MissingPredictions {
        dir: String,
        missing: Vec<String>,
    },
    Format(FormatError),
    Convert(ConvertError),
    Metric(MetricError),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::pyfmt::repr_str;
        match self {
            Self::DuplicateMetrics(names) => write!(
                f,
                "duplicate metric classes in metrics: {}",
                names.join(", ")
            ),
            Self::NoEvalClasses { protocol } => {
                write!(
                    f,
                    "protocol {} declares no eval_classes",
                    repr_str(protocol)
                )
            }
            Self::RepeatedClass { protocol, classes } => {
                let items: Vec<String> = classes.iter().map(i64::to_string).collect();
                let tuple = if items.len() == 1 {
                    format!("({},)", items[0])
                } else {
                    format!("({})", items.join(", "))
                };
                write!(
                    f,
                    "protocol {} repeats a class in {tuple}",
                    repr_str(protocol)
                )
            }
            Self::NoSequences { dataset } => {
                write!(f, "dataset {} has no sequences to score", repr_str(dataset))
            }
            Self::GeometryMismatch {
                metric,
                reads,
                sequence,
                dataset,
                holds,
            } => write!(
                f,
                "{metric} reads {reads}, but sequence {} of dataset {} holds {holds}",
                repr_str(sequence),
                repr_str(dataset)
            ),
            Self::MissingPredictions { dir, missing } => write!(
                f,
                "prediction directory {dir} has no <seq>.txt file for {} sequence(s): {} \
                 (an empty file means no predictions)",
                missing.len(),
                missing.join(", ")
            ),
            Self::Format(err) => err.fmt(f),
            Self::Convert(err) => err.fmt(f),
            Self::Metric(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for EvalError {}

impl EvalError {
    pub fn python_exception(&self) -> &'static str {
        match self {
            Self::Format(err) => err.python_exception(),
            Self::Convert(err) => err.python_exception(),
            Self::Metric(err) => err.python_exception(),
            _ => "ValueError",
        }
    }
}

// CPython: `UnicodeDecodeError` and `json.JSONDecodeError` subclass `ValueError`.
pub fn is_value_error(exception: &str) -> bool {
    matches!(
        exception,
        "ValueError" | "UnicodeDecodeError" | "JSONDecodeError" | "Unsupported"
    )
}

impl From<FormatError> for EvalError {
    fn from(err: FormatError) -> Self {
        Self::Format(err)
    }
}

impl From<ConvertError> for EvalError {
    fn from(err: ConvertError) -> Self {
        Self::Convert(err)
    }
}

impl From<MetricError> for EvalError {
    fn from(err: MetricError) -> Self {
        Self::Metric(err)
    }
}

fn upsert<V>(map: &mut Vec<(String, V)>, key: &str, value: V) {
    match map.iter_mut().find(|(name, _)| name == key) {
        Some((_, slot)) => *slot = value,
        None => map.push((key.to_string(), value)),
    }
}

fn preflight(
    dataset: &MOTDataset,
    pred_dir: &Path,
    shown_dir: &str,
    metrics: &[Box<dyn Metric>],
) -> Result<(), EvalError> {
    let names: Vec<&str> = metrics.iter().map(|metric| metric.name()).collect();
    let mut duplicates: Vec<String> = names
        .iter()
        .filter(|name| names.iter().filter(|other| other == name).count() > 1)
        .map(|name| name.to_string())
        .collect();
    duplicates.sort();
    duplicates.dedup();
    if !duplicates.is_empty() {
        return Err(EvalError::DuplicateMetrics(duplicates));
    }

    let protocol = &dataset.protocol;
    let classes = &protocol.eval_classes;
    if classes.is_empty() {
        return Err(EvalError::NoEvalClasses {
            protocol: protocol.name.clone(),
        });
    }
    let mut seen = classes.clone();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != classes.len() {
        return Err(EvalError::RepeatedClass {
            protocol: protocol.name.clone(),
            classes: classes.clone(),
        });
    }

    if dataset.sequences.is_empty() {
        return Err(EvalError::NoSequences {
            dataset: dataset.name.clone(),
        });
    }

    for metric in metrics {
        let Some(reads) = metric.geometry() else {
            continue;
        };
        for seq in &dataset.sequences {
            if seq.kind() != reads {
                return Err(EvalError::GeometryMismatch {
                    metric: metric.name().to_string(),
                    reads: reads.as_str(),
                    sequence: seq.name().to_string(),
                    dataset: dataset.name.clone(),
                    holds: seq.kind().as_str(),
                });
            }
        }
    }

    let missing = missing_prediction_files(pred_dir, dataset.sequences.iter().map(Sequence::name));
    if !missing.is_empty() {
        return Err(EvalError::MissingPredictions {
            dir: shown_dir.to_string(),
            missing,
        });
    }
    Ok(())
}

fn prediction_file(pred_dir: &Path, name: &str) -> PathBuf {
    py_join(pred_dir, format!("{name}.txt"))
}

pub fn evaluate(
    dataset: &MOTDataset,
    predictions: &Path,
    metrics: &[Box<dyn Metric>],
) -> Result<EvaluationResult, EvalError> {
    // CPython `pathlib.Path` normalizes the path; the messages and the reads use that string.
    let shown_dir = pyfmt::path_str(&predictions.to_string_lossy());
    let pred_dir = PathBuf::from(&shown_dir);
    preflight(dataset, &pred_dir, &shown_dir, metrics)?;
    let multi_class = dataset.protocol.eval_classes.len() > 1;

    let mut box_preds: Vec<(String, Vec<Track>)> = Vec::new();
    let mut mask_preds: Vec<(String, Vec<MaskTrack>)> = Vec::new();
    for seq in &dataset.sequences {
        let path = prediction_file(&pred_dir, seq.name());
        match seq {
            Sequence::Masks(_) => upsert(&mut mask_preds, seq.name(), read_mots(&path)?),
            Sequence::Boxes(_) => {
                upsert(&mut box_preds, seq.name(), read_mot(&path, multi_class)?);
            }
        }
    }
    fn lookup<'a, T>(map: &'a [(String, Vec<T>)], name: &str) -> &'a [T] {
        &map.iter()
            .find(|(seq, _)| seq == name)
            .expect("every sequence's predictions were read")
            .1
    }

    score(dataset, metrics, |seq, cls_id| match seq {
        Sequence::Boxes(gt) => Ok(build_sequence_data(
            gt,
            lookup(&box_preds, &gt.name),
            &dataset.protocol,
            cls_id,
        )?),
        Sequence::Masks(gt) => Ok(build_mask_sequence_data(
            gt,
            lookup(&mask_preds, &gt.name),
            &dataset.protocol,
            cls_id,
        )?),
    })
}

type ClassScores = (Vec<(String, MetricScores)>, Vec<(String, Scores)>);

fn score_class(
    dataset: &MOTDataset,
    metrics: &[Box<dyn Metric>],
    cls_id: i64,
    build: &mut impl FnMut(&Sequence, i64) -> Result<SequenceData, EvalError>,
) -> Result<ClassScores, EvalError> {
    let mut per_sequence: Vec<(String, MetricScores)> = Vec::new();
    let mut by_metric: Vec<Vec<(String, Scores)>> = vec![Vec::new(); metrics.len()];
    for seq in &dataset.sequences {
        let data = build(seq, cls_id)?;
        let mut seq_scores = MetricScores::new();
        for (index, metric) in metrics.iter().enumerate() {
            let scores = metric.eval_sequence(&data)?;
            let declared = scores.declared(metric.fields());
            upsert(&mut by_metric[index], seq.name(), scores);
            if !declared.is_empty() {
                seq_scores.insert(metric.name(), declared);
            }
        }
        upsert(&mut per_sequence, seq.name(), seq_scores);
    }
    let mut raw_combined = Vec::with_capacity(metrics.len());
    for (metric, results) in metrics.iter().zip(&by_metric) {
        let all_res: Vec<&Scores> = results.iter().map(|(_, scores)| scores).collect();
        raw_combined.push((
            metric.name().to_string(),
            metric.combine_sequences(&all_res)?,
        ));
    }
    Ok((per_sequence, raw_combined))
}

fn declared_metrics(raw: &[(String, Scores)], metrics: &[Box<dyn Metric>]) -> MetricScores {
    let mut out = MetricScores::new();
    for ((name, scores), metric) in raw.iter().zip(metrics) {
        out.insert(name.clone(), scores.declared(metric.fields()));
    }
    out
}

fn score(
    dataset: &MOTDataset,
    metrics: &[Box<dyn Metric>],
    mut build: impl FnMut(&Sequence, i64) -> Result<SequenceData, EvalError>,
) -> Result<EvaluationResult, EvalError> {
    let classes = &dataset.protocol.eval_classes;
    let mut scored: Vec<(i64, ClassScores)> = Vec::with_capacity(classes.len());
    for &cls_id in classes {
        scored.push((cls_id, score_class(dataset, metrics, cls_id, &mut build)?));
    }

    if classes.len() == 1 {
        let (_, (per_sequence, raw_combined)) = scored.pop().expect("one class was scored");
        return Ok(EvaluationResult {
            per_sequence,
            combined: declared_metrics(&raw_combined, metrics),
            ..EvaluationResult::default()
        });
    }

    let mut class_averaged = MetricScores::new();
    let mut det_averaged = MetricScores::new();
    for (index, metric) in metrics.iter().enumerate() {
        let raw_by_class: Vec<&Scores> = scored
            .iter()
            .map(|(_, (_, raw_combined))| &raw_combined[index].1)
            .collect();
        class_averaged.insert(
            metric.name(),
            metric
                .combine_classes_class_averaged(&raw_by_class)?
                .declared(metric.fields()),
        );
        det_averaged.insert(
            metric.name(),
            metric
                .combine_classes_det_averaged(&raw_by_class)?
                .declared(metric.fields()),
        );
    }
    let per_class = scored
        .into_iter()
        .map(|(cls_id, (per_sequence, raw_combined))| {
            let combined = declared_metrics(&raw_combined, metrics);
            (
                cls_id,
                ClassResult {
                    per_sequence,
                    combined,
                },
            )
        })
        .collect();
    Ok(EvaluationResult {
        per_class,
        class_averaged,
        det_averaged,
        ..EvaluationResult::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Value;
    use crate::model::{
        FrameConvention, Geometry, GtSequence, Matrix, Protocol, SequenceDataParts,
    };

    struct Stub {
        name: &'static str,
        per_sequence_fields: bool,
    }

    const STUB_FIELDS: &[&str] = &["n", "inputs", "classes"];

    fn float_sum(all_res: &[&Scores], field: &str) -> f64 {
        all_res.iter().map(|res| res.float(field)).sum()
    }

    impl Metric for Stub {
        fn name(&self) -> &'static str {
            self.name
        }

        fn fields(&self) -> &'static [&'static str] {
            STUB_FIELDS
        }

        fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError> {
            let mut scores = Scores::new();
            let n = data.num_gt_dets() as f64;
            let field = if self.per_sequence_fields { "n" } else { "_n" };
            scores.insert(field, Value::Float(n));
            scores.insert("_len", Value::Float(data.name().len() as f64));
            Ok(scores)
        }

        fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
            let field = if self.per_sequence_fields { "n" } else { "_n" };
            let mut scores = Scores::new();
            scores.insert("_len", Value::Float(float_sum(all_res, "_len")));
            scores.insert("n", Value::Float(float_sum(all_res, field)));
            scores.insert("inputs", Value::Float(all_res.len() as f64));
            Ok(scores)
        }

        fn combine_classes_class_averaged(
            &self,
            all_res: &[&Scores],
        ) -> Result<Scores, MetricError> {
            let mut scores = Scores::new();
            scores.insert("classes", Value::Float(all_res.len() as f64));
            scores.insert("n", Value::Float(float_sum(all_res, "_len")));
            Ok(scores)
        }

        fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
            let mut scores = Scores::new();
            scores.insert("n", Value::Float(-float_sum(all_res, "n")));
            Ok(scores)
        }
    }

    fn dataset(names: &[&str], classes: Vec<i64>) -> MOTDataset {
        let sequences = names
            .iter()
            .map(|name| {
                Sequence::Boxes(GtSequence {
                    name: name.to_string(),
                    num_timesteps: 1,
                    tracks: Vec::new(),
                    ignore_regions: Vec::new(),
                })
            })
            .collect();
        MOTDataset {
            name: "stub".into(),
            split: "val".into(),
            sequences,
            protocol: Protocol::new("stub", FrameConvention::new("1-indexed", 1), classes),
        }
    }

    fn view(name: &str, dets: usize) -> SequenceData {
        SequenceData::new(SequenceDataParts {
            name: name.to_string(),
            num_timesteps: 1,
            num_gt_ids: dets,
            num_pred_ids: 0,
            num_gt_dets: dets,
            num_pred_dets: 0,
            gt_ids: vec![(0..dets).collect()],
            pred_ids: vec![Vec::new()],
            pred_confidences: vec![Vec::new()],
            geometry: Geometry::Boxes {
                gt: vec![vec![[0.0, 0.0, 1.0, 1.0]; dets]],
                pred: vec![Vec::new()],
            },
            similarity: vec![Matrix::zeros(dets, 0)],
        })
        .expect("a consistent view")
    }

    fn stubs() -> Vec<Box<dyn Metric>> {
        vec![
            Box::new(Stub {
                name: "B",
                per_sequence_fields: true,
            }),
            Box::new(Stub {
                name: "A",
                per_sequence_fields: false,
            }),
        ]
    }

    fn keys<V>(map: &[(String, V)]) -> Vec<&str> {
        map.iter().map(|(key, _)| key.as_str()).collect()
    }

    fn field_names(scores: &Scores) -> Vec<&str> {
        scores.iter().map(|(name, _)| name).collect()
    }

    #[test]
    fn single_class_keeps_declared_fields_in_dataset_and_metric_order() {
        let data = dataset(&["s2", "seq1"], vec![7]);
        let mut calls = Vec::new();
        let result = score(&data, &stubs(), |seq, cls_id| {
            calls.push((seq.name().to_string(), cls_id));
            Ok(view(seq.name(), if seq.name() == "s2" { 3 } else { 5 }))
        })
        .unwrap();
        assert_eq!(calls, [("s2".to_string(), 7), ("seq1".to_string(), 7)]);
        assert!(!result.is_multi_class());
        assert_eq!(keys(&result.per_sequence), ["s2", "seq1"]);
        let first = &result.per_sequence[0].1;
        let metrics: Vec<&str> = first.iter().map(|(m, _)| m).collect();
        assert_eq!(metrics, ["B"]);
        assert_eq!(field_names(first.get("B").unwrap()), ["n"]);
        assert_eq!(first.get("B").unwrap().float("n"), 3.0);
        let names: Vec<&str> = result.combined.iter().map(|(m, _)| m).collect();
        assert_eq!(names, ["B", "A"]);
        for (_, scores) in result.combined.iter() {
            assert_eq!(field_names(scores), ["n", "inputs"]);
            assert_eq!(scores.float("n"), 8.0);
            assert_eq!(scores.float("inputs"), 2.0);
        }
        assert!(result.per_class.is_empty());
        assert!(result.class_averaged.is_empty() && result.det_averaged.is_empty());
    }

    #[test]
    fn multi_class_combines_raw_class_state_in_eval_classes_order() {
        let data = dataset(&["abc", "de"], vec![2, 1]);
        let result = score(&data, &stubs(), |seq, cls_id| {
            Ok(view(seq.name(), cls_id as usize))
        })
        .unwrap();
        assert!(result.is_multi_class());
        assert!(result.per_sequence.is_empty() && result.combined.is_empty());
        let classes: Vec<i64> = result.per_class.iter().map(|(c, _)| *c).collect();
        assert_eq!(classes, [2, 1]);
        let class_two = &result.per_class[0].1;
        assert_eq!(keys(&class_two.per_sequence), ["abc", "de"]);
        assert_eq!(class_two.combined.get("B").unwrap().float("n"), 4.0);
        assert_eq!(
            field_names(class_two.combined.get("A").unwrap()),
            ["n", "inputs"]
        );
        for metric in ["B", "A"] {
            let averaged = result.class_averaged.get(metric).unwrap();
            assert_eq!(field_names(averaged), ["classes", "n"]);
            assert_eq!(averaged.float("n"), 10.0);
            assert_eq!(averaged.float("classes"), 2.0);
            assert_eq!(result.det_averaged.get(metric).unwrap().float("n"), -6.0);
        }
    }

    #[test]
    fn a_repeated_sequence_name_keeps_one_entry_like_a_python_dict() {
        let data = dataset(&["x", "y", "x"], vec![1]);
        let mut dets = [1usize, 2, 4].into_iter();
        let result = score(&data, &stubs(), |seq, _| {
            Ok(view(seq.name(), dets.next().unwrap()))
        })
        .unwrap();
        assert_eq!(keys(&result.per_sequence), ["x", "y"]);
        assert_eq!(
            result.per_sequence[0].1.get("B").unwrap().float("n"),
            4.0,
            "the later sequence replaces the value in place"
        );
        let combined = result.combined.get("B").unwrap();
        assert_eq!(combined.float("inputs"), 2.0);
        assert_eq!(combined.float("n"), 6.0);
    }

    #[test]
    fn a_build_error_stops_scoring() {
        let data = dataset(&["x", "y"], vec![1]);
        let mut built = 0;
        let err = score(&data, &stubs(), |seq, _| {
            built += 1;
            if seq.name() == "x" {
                Err(EvalError::Convert(ConvertError::OverlappingGtMasks {
                    sequence: "x".into(),
                    timestep: 3,
                }))
            } else {
                Ok(view(seq.name(), 1))
            }
        })
        .unwrap_err();
        assert_eq!(built, 1);
        assert_eq!(
            err.to_string(),
            "sequence 'x': overlapping GT masks at timestep 3"
        );
    }
}
