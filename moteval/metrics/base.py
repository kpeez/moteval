"""Metric ABC shared by every metric.

A metric declares its ``fields``, scores one `SequenceData` at a time, and knows
how to combine per-sequence results and per-class results (class-averaged and
detection-averaged). Values are plain field->float dicts so combining is generic.
"""

from abc import ABC, abstractmethod

from moteval.data.model import SequenceData

Scores = dict[str, float]


class Metric(ABC):
    fields: tuple[str, ...]

    @abstractmethod
    def eval_sequence(self, data: SequenceData) -> Scores: ...

    @abstractmethod
    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores: ...

    @abstractmethod
    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores: ...

    @abstractmethod
    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores: ...

    def _combine_sum(self, all_res: dict[str, Scores], field: str) -> float:
        return sum(res[field] for res in all_res.values())
