"""Typed evaluation results and their stable JSON and CSV export shapes.

A single-class run fills ``per_sequence`` and ``combined``. A multi-class run
fills ``per_class``, ``class_averaged`` and ``det_averaged`` instead, and leaves
``per_sequence`` and ``combined`` empty. The exports follow the same split, so
single-class JSON and CSV keep their original shape:

- JSON: ``dataset``, ``split``, ``per_sequence``, ``combined``. A multi-class run
  adds ``per_class`` (class id as a string -> ``per_sequence`` + ``combined``),
  ``class_averaged`` and ``det_averaged``.
- CSV: single-class columns are ``seq, metric, field, value``. Multi-class columns
  are ``class, seq, metric, field, value``: one block per class id (each sequence,
  then ``COMBINED``), then ``class_averaged`` and ``det_averaged`` rows with
  ``seq = COMBINED``. An array field is reduced to its mean.

The Rust extension writes both exports (``rust/moteval/src/results.rs``), so the CLI and
the Python API give the same bytes.
"""

import json
from dataclasses import dataclass, field
from typing import TypeAlias, TypedDict, cast

from moteval import _moteval
from moteval._rust import call
from moteval.metrics.base import Scores

# metric name -> field -> value
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
    # Present only for multi-class runs.
    per_class: dict[str, JsonClassResult]
    class_averaged: JsonScores
    det_averaged: JsonScores


@dataclass(frozen=True)
class ClassResult:
    """Scores for one class of a multi-class run, shaped like a single-class run."""

    per_sequence: dict[str, MetricScores]
    combined: MetricScores


@dataclass(frozen=True)
class EvaluationResult:
    """Scores for one evaluation run.

    ``per_sequence`` maps sequence name -> metric name -> field -> value;
    ``combined`` maps metric name -> field -> value across all sequences. They
    hold the scores of a single-class run and are empty for a multi-class run.

    ``per_class`` maps class id -> `ClassResult` (the same two mappings for that
    class). ``class_averaged`` and ``det_averaged`` map metric name -> field ->
    value, combined across classes by the metric's class-averaged and
    detection-averaged combiners. All three are empty for a single-class run.

    Every mapping holds only each metric's declared ``Metric.fields``. A metric
    whose per-sequence state holds none of its fields (TrackMAP, whose AP pools
    detections across sequences) is absent from the per-sequence mappings.
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
    """The stable JSON export text: ``moteval run --out-json`` writes exactly this."""
    return call(_moteval.results_to_json, result, dataset, split)


def to_json_dict(result: EvaluationResult, *, dataset: str, split: str) -> JsonResult:
    """The stable JSON export as a dict (`to_json` parsed; NaN stays ``float("nan")``)."""
    return cast(JsonResult, json.loads(to_json(result, dataset=dataset, split=split)))


def to_csv(result: EvaluationResult) -> str:
    """The stable long-form CSV export text: ``moteval run --out-csv`` writes exactly
    this (``\\r\\n`` line ends, as Python's `csv.writer` writes)."""
    return call(_moteval.results_to_csv, result)
