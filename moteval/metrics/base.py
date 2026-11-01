"""Metric ABC shared by every metric, and the base of the built-in metrics.

A metric declares its ``fields`` and the ``geometry`` it reads, scores one
`SequenceData` at a time, and knows how to combine per-sequence results and
per-class results (class-averaged and detection-averaged). Values are field->float
or field->19-alpha-array dicts so combining is generic.

The six built-in metrics (`HOTA`, `CLEAR`, `Identity`, `Count`, `TrackMAP`, `JAndF`)
run in the Rust extension (``src/metrics/``): `RustMetric` sends the
`SequenceData` or the score dicts across and returns plain ``float`` and float64 numpy
values.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from moteval import _moteval
from moteval._rust import call
from moteval.data.model import SequenceData

Scores = dict[str, float | np.ndarray]

Geometry = Literal["boxes", "masks"] | None


class Metric(ABC):
    fields: tuple[str, ...]
    """The public result fields: the only keys `evaluate` reports for this metric.

    Combined results hold exactly these fields; per-sequence results hold a subset.
    ``eval_sequence`` and ``combine_sequences`` may also return private state
    (TrackMAP's match arrays and ``_num_dt_*`` weights) that only the combine
    methods consume. A metric whose per-sequence state holds none of its fields
    (TrackMAP: AP pools detections across sequences, so ``eval_sequence`` returns
    only private match state) has no per-sequence entry in the results.
    """

    geometry: Geometry = None
    """The geometry ``eval_sequence`` reads besides the precomputed similarity.

    ``"boxes"`` (TrackMAP), ``"masks"`` (J&F), or ``None`` for a metric that reads
    only similarity. `evaluate` rejects a dataset of the other kind before it
    scores anything.
    """

    @abstractmethod
    def eval_sequence(self, data: SequenceData) -> Scores: ...

    @abstractmethod
    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores: ...

    @abstractmethod
    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores: ...

    @abstractmethod
    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores: ...


def rust_metric_info(name: str) -> tuple[tuple[str, ...], Geometry]:
    """The ``fields`` and ``geometry`` of the built-in metric ``name``, from Rust."""
    fields, geometry = _moteval.metric_info(name)
    if geometry not in ("boxes", "masks", None):
        raise ValueError(f"metric {name} reads unknown geometry {geometry!r}")
    return tuple(fields), geometry


class RustMetric(Metric):
    """A built-in metric computed by the Rust extension.

    ``rust_name`` selects the Rust metric; it is also the metric's key in results.
    """

    rust_name: str

    def eval_sequence(self, data: SequenceData) -> Scores:
        return call(_moteval.metric_eval_sequence, self.rust_name, data)

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        return call(_moteval.metric_combine, self.rust_name, "combine_sequences", all_res)

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return call(
            _moteval.metric_combine, self.rust_name, "combine_classes_class_averaged", all_res
        )

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return call(
            _moteval.metric_combine, self.rust_name, "combine_classes_det_averaged", all_res
        )
