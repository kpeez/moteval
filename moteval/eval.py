"""Top-level evaluation: score a loaded `MOTDataset` against prediction files.

`evaluate(dataset, predictions, metrics)` reads ``<seq>.txt`` MOTChallenge (or
MOTS) predictions from a directory and returns typed results. Every sequence
needs a file; an empty file means no predictions. A missing file raises
``ValueError`` before any sequence is scored. It is the whole
public scoring seam: any `MOTDataset` — built-in benchmark, generic-layout load,
or hand-constructed — evaluates through it.

Each class in ``protocol.eval_classes`` is scored on its own: one `SequenceData`
view per sequence, every metric, then the sequences combined for that class. A
single-class run reports those scores as ``per_sequence`` and ``combined``. A
multi-class run reports them under ``per_class`` and also combines the classes
in the two ways TrackEval does: ``class_averaged`` and ``det_averaged``. Box
predictions give their class in column 8, which is read only for multi-class
protocols.

Results hold only each metric's declared ``fields``. The raw per-sequence and
combined state, which can carry private keys (TrackMAP's match arrays and
``_num_dt_*`` weights), stays inside `evaluate`, where the class combiners use it.
"""

from collections.abc import Sequence
from pathlib import Path

from moteval.data.convert import build_mask_sequence_data, build_sequence_data
from moteval.data.model import MaskGtSequence, MOTDataset
from moteval.formats import MaskTrack, Track, read_mot, read_mots
from moteval.metrics.base import Metric, Scores
from moteval.results import ClassResult, EvaluationResult, MetricScores


def _declared(scores: Scores, fields: tuple[str, ...]) -> Scores:
    return {field: value for field, value in scores.items() if field in fields}


def _declared_metrics(raw: dict[str, Scores], metrics: dict[str, Metric]) -> MetricScores:
    return {name: _declared(raw[name], metric.fields) for name, metric in metrics.items()}


def _score_class(
    dataset: MOTDataset,
    box_preds: dict[str, tuple[Track, ...]],
    mask_preds: dict[str, tuple[MaskTrack, ...]],
    metrics: dict[str, Metric],
    cls_id: int,
) -> tuple[dict[str, MetricScores], dict[str, Scores]]:
    """Score one class: declared per-sequence scores and raw combined state per metric."""
    per_sequence: dict[str, MetricScores] = {}
    by_metric: dict[str, dict[str, Scores]] = {name: {} for name in metrics}
    for seq in dataset.sequences:
        if isinstance(seq, MaskGtSequence):
            data = build_mask_sequence_data(seq, mask_preds[seq.name], dataset.protocol, cls_id)
        else:
            data = build_sequence_data(seq, box_preds[seq.name], dataset.protocol, cls_id)
        seq_scores: MetricScores = {}
        for name, metric in metrics.items():
            scores = metric.eval_sequence(data)
            by_metric[name][seq.name] = scores
            declared = _declared(scores, metric.fields)
            # A metric with no per-sequence fields (TrackMAP) has no per-sequence entry.
            if declared:
                seq_scores[name] = declared
        per_sequence[seq.name] = seq_scores

    raw_combined = {
        name: metric.combine_sequences(by_metric[name]) for name, metric in metrics.items()
    }
    return per_sequence, raw_combined


def evaluate(
    dataset: MOTDataset,
    predictions: str | Path,
    metrics: Sequence[Metric],
) -> EvaluationResult:
    pred_dir = Path(predictions)
    names = [type(metric).__name__ for metric in metrics]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate metric classes in metrics: {', '.join(duplicates)}")
    by_name = dict(zip(names, metrics, strict=True))

    classes = dataset.protocol.eval_classes
    if not classes:
        raise ValueError(f"protocol {dataset.protocol.name!r} declares no eval_classes")
    if len(set(classes)) != len(classes):
        raise ValueError(f"protocol {dataset.protocol.name!r} repeats a class in {classes}")
    multi_class = len(classes) > 1

    # Every sequence needs a prediction file, as in TrackEval; an empty file means
    # no predictions. Name every missing sequence before scoring any of them.
    missing = [
        seq.name for seq in dataset.sequences if not (pred_dir / f"{seq.name}.txt").is_file()
    ]
    if missing:
        raise ValueError(
            f"prediction directory {pred_dir} has no <seq>.txt file for {len(missing)} "
            f"sequence(s): {', '.join(missing)} (an empty file means no predictions)"
        )

    # Read each prediction file once; every class view filters the same rows.
    box_preds: dict[str, tuple[Track, ...]] = {}
    mask_preds: dict[str, tuple[MaskTrack, ...]] = {}
    for seq in dataset.sequences:
        pred_file = pred_dir / f"{seq.name}.txt"
        if isinstance(seq, MaskGtSequence):
            mask_preds[seq.name] = tuple(read_mots(pred_file))
        else:
            box_preds[seq.name] = tuple(read_mot(pred_file, class_column=multi_class))

    scored = {
        cls_id: _score_class(dataset, box_preds, mask_preds, by_name, cls_id) for cls_id in classes
    }

    if not multi_class:
        per_sequence, raw_combined = scored[classes[0]]
        return EvaluationResult(
            per_sequence=per_sequence, combined=_declared_metrics(raw_combined, by_name)
        )

    # The class combiners take the RAW per-class combined state: TrackMAP's
    # `combine_classes_det_averaged` weights by the private `_num_dt_*` fields.
    # Classes enter in `eval_classes` order, as upstream's class list does.
    class_averaged: MetricScores = {}
    det_averaged: MetricScores = {}
    for name, metric in by_name.items():
        raw_by_class = {str(cls_id): scored[cls_id][1][name] for cls_id in classes}
        class_averaged[name] = _declared(
            metric.combine_classes_class_averaged(raw_by_class), metric.fields
        )
        det_averaged[name] = _declared(
            metric.combine_classes_det_averaged(raw_by_class), metric.fields
        )
    per_class = {
        cls_id: ClassResult(
            per_sequence=per_sequence, combined=_declared_metrics(raw_combined, by_name)
        )
        for cls_id, (per_sequence, raw_combined) in scored.items()
    }
    return EvaluationResult(
        per_sequence={},
        combined={},
        per_class=per_class,
        class_averaged=class_averaged,
        det_averaged=det_averaged,
    )
