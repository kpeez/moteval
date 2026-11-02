import json
from dataclasses import dataclass, field
from typing import TypeAlias, TypedDict, cast

from moteval import _moteval
from moteval._rust import call
from moteval.metrics.base import Scores

MetricScores = dict[str, Scores]
JsonScalar: TypeAlias = str | int | float
JsonValue: TypeAlias = JsonScalar | list["JsonValue"]
JsonScores: TypeAlias = dict[str, dict[str, JsonValue]]


class JsonClassResult(TypedDict):
    per_sequence: dict[str, JsonScores]
    combined: JsonScores


class _JsonResultBase(TypedDict):
    dataset: str
    split: str
    per_sequence: dict[str, JsonScores]
    combined: JsonScores


class JsonResult(_JsonResultBase, total=False):
    per_class: dict[str, JsonClassResult]
    class_averaged: JsonScores
    det_averaged: JsonScores


@dataclass(frozen=True)
class ClassResult:
    per_sequence: dict[str, MetricScores]
    combined: MetricScores


@dataclass(frozen=True)
class EvaluationResult:
    """A single-class run fills ``per_sequence`` and ``combined``; a multi-class run fills
    ``per_class``, ``class_averaged`` and ``det_averaged`` instead. Each holds only the
    metrics' ``fields``, and TrackMAP has no per-sequence entry.
    """

    per_sequence: dict[str, MetricScores]
    combined: MetricScores
    per_class: dict[int, ClassResult] = field(default_factory=dict)
    class_averaged: MetricScores = field(default_factory=dict)
    det_averaged: MetricScores = field(default_factory=dict)

    @property
    def is_multi_class(self) -> bool:
        return bool(self.per_class)


def to_json(result: EvaluationResult, *, dataset: str, split: str) -> str:
    return call(_moteval.results_to_json, result, dataset, split)


def to_json_dict(result: EvaluationResult, *, dataset: str, split: str) -> JsonResult:
    return cast(JsonResult, json.loads(to_json(result, dataset=dataset, split=split)))


def to_csv(result: EvaluationResult) -> str:
    return call(_moteval.results_to_csv, result)
