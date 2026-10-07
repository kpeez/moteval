"""Top-level evaluation: score a loaded `MOTDataset` against prediction files.

`evaluate(dataset, predictions, metrics)` reads ``<seq>.txt`` MOTChallenge (or
MOTS) predictions from a directory and returns typed per-sequence and combined
results. It is the whole public scoring seam: any `MOTDataset` — built-in
benchmark, generic-layout load, or hand-constructed — evaluates through it.

Results hold only each metric's declared ``fields``. The raw per-sequence and
combined state, which can carry private keys (TrackMAP's match arrays and
``_num_dt_*`` weights), stays inside `evaluate`.
"""

from collections.abc import Sequence
from pathlib import Path

from moteval.data.convert import build_mask_sequence_data, build_sequence_data
from moteval.data.model import MaskGtSequence, MOTDataset
from moteval.formats import read_mot, read_mots
from moteval.metrics.base import Metric, Scores
from moteval.results import EvaluationResult, MetricScores


def _declared(scores: Scores, fields: tuple[str, ...]) -> Scores:
    return {field: value for field, value in scores.items() if field in fields}


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

    protocol = dataset.protocol
    if len(protocol.eval_classes) != 1:
        raise ValueError("evaluate() supports single-class protocols; multi-class is a later slice")
    (cls_id,) = protocol.eval_classes

    per_sequence: dict[str, MetricScores] = {}
    by_metric: dict[str, dict[str, Scores]] = {name: {} for name in names}

    for seq in dataset.sequences:
        pred_file = pred_dir / f"{seq.name}.txt"
        if isinstance(seq, MaskGtSequence):
            mask_preds = tuple(read_mots(pred_file)) if pred_file.is_file() else ()
            data = build_mask_sequence_data(seq, mask_preds, protocol, cls_id)
        else:
            box_preds = tuple(read_mot(pred_file)) if pred_file.is_file() else ()
            data = build_sequence_data(seq, box_preds, protocol, cls_id)
        seq_scores: MetricScores = {}
        for name, metric in zip(names, metrics, strict=True):
            scores = metric.eval_sequence(data)
            by_metric[name][seq.name] = scores
            declared = _declared(scores, metric.fields)
            # A metric with no per-sequence fields (TrackMAP) has no per-sequence entry.
            if declared:
                seq_scores[name] = declared
        per_sequence[seq.name] = seq_scores

    # Raw combined state, private keys included. Combining classes needs it:
    # TrackMAP's `combine_classes_det_averaged` weights by `_num_dt_*`.
    raw_combined = {
        name: metric.combine_sequences(by_metric[name])
        for name, metric in zip(names, metrics, strict=True)
    }
    combined = {
        name: _declared(raw_combined[name], metric.fields)
        for name, metric in zip(names, metrics, strict=True)
    }
    return EvaluationResult(per_sequence=per_sequence, combined=combined)
