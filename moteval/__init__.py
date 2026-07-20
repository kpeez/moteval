"""moteval: a from-scratch TrackEval rewrite. Public API surface.

`evaluate(dataset, predictions, metrics)` scores a loaded `MOTDataset` against a
directory of ``<seq>.txt`` MOTChallenge predictions and returns typed per-sequence
and combined results.
"""

from collections.abc import Sequence
from pathlib import Path

from moteval.benchmarks import animaltrack as _animaltrack  # noqa: F401  (registers animaltrack)
from moteval.benchmarks import bft as _bft  # noqa: F401  (registers bft)
from moteval.benchmarks import chimpact as _chimpact  # noqa: F401  (registers chimpact)
from moteval.benchmarks import dancetrack as _dancetrack  # noqa: F401  (registers dancetrack)
from moteval.benchmarks import gmot40 as _gmot40  # noqa: F401  (registers gmot40)
from moteval.benchmarks import mots20 as _mots20  # noqa: F401  (registers mots20)
from moteval.benchmarks import panaf500 as _panaf500  # noqa: F401  (registers panaf500)
from moteval.benchmarks import sportsmot as _sportsmot  # noqa: F401  (registers sportsmot)
from moteval.benchmarks import uavdt as _uavdt  # noqa: F401  (registers uavdt)
from moteval.data.convert import build_mask_sequence_data, build_sequence_data
from moteval.data.model import (
    FrameConvention,
    GtSequence,
    MaskGtSequence,
    MOTDataset,
    SequenceData,
)
from moteval.data.protocol import Protocol
from moteval.data.registry import load_dataset, register_dataset
from moteval.formats import MaskTrack, Track, read_mot, read_mots
from moteval.metrics.base import Metric, Scores
from moteval.metrics.clear import CLEAR
from moteval.metrics.count import Count
from moteval.metrics.hota import HOTA
from moteval.metrics.identity import Identity
from moteval.metrics.jf import JAndF
from moteval.metrics.track_map import TrackMAP
from moteval.results import EvaluationResult, MetricScores

__all__ = [
    "CLEAR",
    "HOTA",
    "Count",
    "EvaluationResult",
    "FrameConvention",
    "GtSequence",
    "Identity",
    "JAndF",
    "MOTDataset",
    "MaskGtSequence",
    "MaskTrack",
    "Metric",
    "Protocol",
    "SequenceData",
    "Track",
    "TrackMAP",
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
            seq_scores[name] = scores
            by_metric[name][seq.name] = scores
        per_sequence[seq.name] = seq_scores

    combined = {
        name: metric.combine_sequences(by_metric[name])
        for name, metric in zip(names, metrics, strict=True)
    }
    return EvaluationResult(per_sequence=per_sequence, combined=combined)
