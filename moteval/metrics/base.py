"""Metric ABC shared by every metric.

A metric declares its ``fields`` and the ``geometry`` it reads, scores one
`SequenceData` at a time, and knows how to combine per-sequence results and
per-class results (class-averaged and detection-averaged). Values are field->float
or field->19-alpha-array dicts so combining is generic.
"""

import functools
import operator
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Literal

import numpy as np

from moteval.data.model import SequenceData

Scores = dict[str, float | np.ndarray]


def _plain_sum(values: Iterable[float | np.ndarray]) -> float | np.ndarray:
    """Add ``values`` left to right from 0, as upstream's ``sum`` over ``np.float64`` does.

    Builtin ``sum`` over exact Python floats is compensated on Python 3.12+, so it would round
    differently from upstream for fields that a metric holds as plain floats.
    """
    return functools.reduce(operator.add, values, 0)


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

    geometry: Literal["boxes", "masks"] | None = None
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

    def _combine_sum(self, all_res: dict[str, Scores], field: str) -> float | np.ndarray:
        return _plain_sum(res[field] for res in all_res.values())

    def _combine_weighted_av(
        self,
        all_res: dict[str, Scores],
        field: str,
        comb_res: Mapping[str, float | np.ndarray],
        weight_field: str,
    ) -> float | np.ndarray:
        weighted_sum = _plain_sum(res[field] * res[weight_field] for res in all_res.values())
        return weighted_sum / np.maximum(1.0, comb_res[weight_field])
