"""moteval: a from-scratch TrackEval rewrite. Public API surface.

`evaluate(dataset, predictions, metrics)` scores a loaded `MOTDataset` against a
directory of ``<seq>.txt`` MOTChallenge predictions and returns typed per-sequence
and combined results.
"""

from collections.abc import Sequence
from pathlib import Path

from moteval.benchmarks import toy as _toy  # noqa: F401  (registers the toy dataset)
from moteval.benchmarks.base import load_dataset, register_dataset
from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention, GtSequence, MOTDataset, SequenceData
from moteval.formats.mot_txt import read_mot
from moteval.metrics.base import Metric
from moteval.metrics.count import Count
from moteval.results import EvaluationResult

__all__ = [
    "Count",
    "EvaluationResult",
    "FrameConvention",
    "GtSequence",
    "MOTDataset",
    "Metric",
    "SequenceData",
    "evaluate",
    "load_dataset",
    "register_dataset",
]


def evaluate(
    dataset: MOTDataset,
    predictions: str | Path,
    metrics: Sequence[Metric],
) -> EvaluationResult:
    pred_dir = Path(predictions)
    per_sequence: dict[str, dict[str, dict[str, float]]] = {}
    by_metric: dict[Metric, dict[str, dict[str, float]]] = {m: {} for m in metrics}

    for seq in dataset.sequences:
        pred_file = pred_dir / f"{seq.name}.txt"
        pred_tracks = tuple(read_mot(pred_file)) if pred_file.is_file() else ()
        data = build_sequence_data(seq, pred_tracks, dataset.frame_convention)
        seq_scores: dict[str, dict[str, float]] = {}
        for metric in metrics:
            scores = metric.eval_sequence(data)
            seq_scores[type(metric).__name__] = scores
            by_metric[metric][seq.name] = scores
        per_sequence[seq.name] = seq_scores

    combined = {
        type(metric).__name__: metric.combine_sequences(by_metric[metric]) for metric in metrics
    }
    return EvaluationResult(per_sequence=per_sequence, combined=combined)
