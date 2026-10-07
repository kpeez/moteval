"""Metric ABC shared by every metric.

A metric declares its ``fields``, scores one `SequenceData` at a time, and knows
how to combine per-sequence results and per-class results (class-averaged and
detection-averaged). Values are field->float or field->19-alpha-array dicts so
combining is generic.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

from moteval.data.model import SequenceData

Scores = dict[str, float | np.ndarray]


class Metric(ABC):
    fields: tuple[str, ...]
    """The public result fields: the only keys `evaluate` reports for this metric.

    Combined results hold exactly these fields; per-sequence results hold a subset.
    ``eval_sequence`` and ``combine_sequences`` may also return private state
    (TrackMAP's match arrays and ``_num_dt_*`` weights) that only the combine
    methods consume. A metric with no per-sequence field (TrackMAP) has no
    per-sequence entry in the results.
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
        return sum(res[field] for res in all_res.values())

    def _combine_weighted_av(
        self,
        all_res: dict[str, Scores],
        field: str,
        comb_res: Mapping[str, float | np.ndarray],
        weight_field: str,
    ) -> float | np.ndarray:
        weighted_sum = sum(res[field] * res[weight_field] for res in all_res.values())
        return weighted_sum / np.maximum(1.0, comb_res[weight_field])
