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
    geometry: Geometry = None

    @abstractmethod
    def eval_sequence(self, data: SequenceData) -> Scores: ...

    @abstractmethod
    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores: ...

    @abstractmethod
    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores: ...

    @abstractmethod
    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores: ...


def rust_metric_info(name: str) -> tuple[tuple[str, ...], Geometry]:
    fields, geometry = _moteval.metric_info(name)
    if geometry not in ("boxes", "masks", None):
        raise ValueError(f"metric {name} reads unknown geometry {geometry!r}")
    return tuple(fields), geometry


class RustMetric(Metric):
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
