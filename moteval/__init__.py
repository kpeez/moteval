"""moteval: a from-scratch TrackEval rewrite. Public API surface.

`evaluate(dataset, predictions, metrics)` scores a loaded `MOTDataset` against a
directory of ``<seq>.txt`` MOTChallenge predictions and returns typed per-sequence
and combined results.
"""

from collections.abc import Sequence
from pathlib import Path

from moteval.benchmarks import animaltrack as _animaltrack  # noqa: F401  (registers animaltrack)
from moteval.benchmarks import bft as _bft  # noqa: F401  (registers bft)
from moteval.benchmarks import dancetrack as _dancetrack  # noqa: F401  (registers dancetrack)
from moteval.benchmarks import gmot40 as _gmot40  # noqa: F401  (registers gmot40)
from moteval.benchmarks import panaf500 as _panaf500  # noqa: F401  (registers panaf500)
from moteval.benchmarks import sportsmot as _sportsmot  # noqa: F401  (registers sportsmot)
from moteval.benchmarks import toy as _toy  # noqa: F401  (registers the toy dataset)
from moteval.benchmarks.base import load_dataset, register_dataset
from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention, GtSequence, MOTDataset, SequenceData
from moteval.data.protocol import Protocol
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
    "Protocol",
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
    names = [type(metric).__name__ for metric in metrics]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate metric classes in metrics: {', '.join(duplicates)}")

    protocol = dataset.protocol
    if len(protocol.eval_classes) != 1:
        raise ValueError("evaluate() supports single-class protocols; multi-class is a later slice")
    (cls_id,) = protocol.eval_classes

    per_sequence: dict[str, dict[str, dict[str, float]]] = {}
    by_metric: dict[str, dict[str, dict[str, float]]] = {name: {} for name in names}

    for seq in dataset.sequences:
        pred_file = pred_dir / f"{seq.name}.txt"
        pred_tracks = tuple(read_mot(pred_file)) if pred_file.is_file() else ()
        data = build_sequence_data(seq, pred_tracks, protocol, cls_id)
        seq_scores: dict[str, dict[str, float]] = {}
        for name, metric in zip(names, metrics, strict=True):
            scores = metric.eval_sequence(data)
            seq_scores[name] = scores
            by_metric[name][seq.name] = scores
        per_sequence[seq.name] = seq_scores

    combined = {
        name: metric.combine_sequences(by_metric[name])
        for name, metric in zip(names, metrics, strict=True)
    }
    return EvaluationResult(per_sequence=per_sequence, combined=combined)
