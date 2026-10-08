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
  ``seq = COMBINED``.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeAlias, TypedDict, cast

import numpy as np

from moteval.metrics.base import Scores

# metric name -> field -> value
MetricScores = dict[str, Scores]
JsonScalar: TypeAlias = str | int | float
JsonValue: TypeAlias = JsonScalar | list["JsonValue"]
JsonScores: TypeAlias = dict[str, dict[str, JsonValue]]
CsvValue: TypeAlias = int | float
CsvRow: TypeAlias = tuple[str, str, str, CsvValue] | tuple[str, str, str, str, CsvValue]

COMBINED = "COMBINED"
CLASS_AVERAGED = "class_averaged"
DET_AVERAGED = "det_averaged"


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


def _to_json_value(value: float | np.ndarray) -> JsonValue:
    if isinstance(value, np.ndarray):
        return cast(JsonValue, value.tolist())
    if isinstance(value, np.generic):
        return cast(int | float, value.item())
    return value


def _json_scores(scores: MetricScores) -> JsonScores:
    return {
        metric: {name: _to_json_value(value) for name, value in fields.items()}
        for metric, fields in scores.items()
    }


def _json_per_sequence(per_sequence: dict[str, MetricScores]) -> dict[str, JsonScores]:
    return {sequence: _json_scores(scores) for sequence, scores in per_sequence.items()}


def to_json_dict(result: EvaluationResult, *, dataset: str, split: str) -> JsonResult:
    """Convert an evaluation result to the stable JSON export schema."""
    exported: JsonResult = {
        "dataset": dataset,
        "split": split,
        "per_sequence": _json_per_sequence(result.per_sequence),
        "combined": _json_scores(result.combined),
    }
    if result.is_multi_class:
        exported["per_class"] = {
            str(cls_id): {
                "per_sequence": _json_per_sequence(class_result.per_sequence),
                "combined": _json_scores(class_result.combined),
            }
            for cls_id, class_result in result.per_class.items()
        }
        exported["class_averaged"] = _json_scores(result.class_averaged)
        exported["det_averaged"] = _json_scores(result.det_averaged)
    return exported


def _to_csv_value(value: float | np.ndarray) -> CsvValue:
    scalar = np.mean(value) if isinstance(value, np.ndarray) else value
    if isinstance(scalar, np.generic):
        return cast(int | float, scalar.item())
    return scalar


def _scored_rows(
    per_sequence: dict[str, MetricScores], combined: MetricScores
) -> Iterator[tuple[str, MetricScores]]:
    yield from per_sequence.items()
    yield COMBINED, combined


def iter_scored_rows(result: EvaluationResult) -> Iterator[tuple[tuple[str, ...], MetricScores]]:
    """Yield ``(row labels, scores)`` in export order.

    Labels are ``(seq,)`` for a single-class run and ``(class, seq)`` for a
    multi-class run; `row_label_names` names them.
    """
    if not result.is_multi_class:
        for sequence, scores in _scored_rows(result.per_sequence, result.combined):
            yield (sequence,), scores
        return
    for cls_id, class_result in result.per_class.items():
        for sequence, scores in _scored_rows(class_result.per_sequence, class_result.combined):
            yield (str(cls_id), sequence), scores
    yield (CLASS_AVERAGED, COMBINED), result.class_averaged
    yield (DET_AVERAGED, COMBINED), result.det_averaged


def row_label_names(result: EvaluationResult) -> tuple[str, ...]:
    """Name the labels `iter_scored_rows` yields: ``("seq",)`` or ``("class", "seq")``."""
    return ("class", "seq") if result.is_multi_class else ("seq",)


def csv_header(result: EvaluationResult) -> tuple[str, ...]:
    """The CSV column names for ``result``; see the module docstring."""
    return (*row_label_names(result), "metric", "field", "value")


def iter_csv_rows(result: EvaluationResult) -> Iterator[CsvRow]:
    """Yield the stable long-form CSV rows, with array fields reduced to their mean."""
    for labels, scores in iter_scored_rows(result):
        for metric, fields in scores.items():
            for field_name, value in fields.items():
                yield cast(CsvRow, (*labels, metric, field_name, _to_csv_value(value)))
