"""moteval: a from-scratch TrackEval rewrite. Public API surface.

Three ways in, all converging on `evaluate(dataset, predictions, metrics)`:

- `load_dataset(name, root=None, split=None)` — a built-in benchmark by name.
- `load_motchallenge(root, split)` / `load_mots(root, split)` — any directory in
  the standard box/mask layout, no registration needed.
- Construct a `MOTDataset` yourself from any source format.
"""

from moteval.benchmarks import load_dataset
from moteval.benchmarks.motchallenge import load_motchallenge
from moteval.benchmarks.mots20 import load_mots
from moteval.data.model import (
    FrameConvention,
    GtSequence,
    MaskGtSequence,
    MOTDataset,
    SequenceData,
)
from moteval.data.protocol import Protocol
from moteval.eval import evaluate
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
    "MetricScores",
    "Protocol",
    "Scores",
    "SequenceData",
    "Track",
    "TrackMAP",
    "evaluate",
    "load_dataset",
    "load_motchallenge",
    "load_mots",
    "read_mot",
    "read_mots",
]
