"""Typed evaluation results: per-sequence and combined metric scores."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias, TypedDict, cast

import numpy as np

from moteval.metrics.base import Scores

# metric name -> field -> value
MetricScores = dict[str, Scores]
JsonScalar: TypeAlias = str | int | float
JsonValue: TypeAlias = JsonScalar | list["JsonValue"]
JsonScores: TypeAlias = dict[str, dict[str, JsonValue]]
CsvRow: TypeAlias = tuple[str, str, str, int | float]


class JsonResult(TypedDict):
    dataset: str
    split: str
    per_sequence: dict[str, JsonScores]
    combined: JsonScores


@dataclass(frozen=True)
class EvaluationResult:
    """Scores for one evaluation run.

    ``per_sequence`` maps sequence name -> metric name -> field -> value;
    ``combined`` maps metric name -> field -> value across all sequences.
    """

    per_sequence: dict[str, MetricScores]
    combined: MetricScores


def _to_json_value(value: float | np.ndarray) -> JsonValue:
    if isinstance(value, np.ndarray):
        array = cast(np.ndarray, value)
        return cast(JsonValue, array.tolist())
    if isinstance(value, np.generic):
        return cast(int | float, value.item())
    return value


def _json_scores(scores: MetricScores) -> JsonScores:
    return {
        metric: {field: _to_json_value(value) for field, value in fields.items()}
        for metric, fields in scores.items()
    }


def to_json_dict(result: EvaluationResult, *, dataset: str, split: str) -> JsonResult:
    """Convert an evaluation result to the stable JSON export schema."""
    return {
        "dataset": dataset,
        "split": split,
        "per_sequence": {
            sequence: _json_scores(scores) for sequence, scores in result.per_sequence.items()
        },
        "combined": _json_scores(result.combined),
    }


def _to_csv_value(value: float | np.ndarray) -> int | float:
    scalar = np.mean(value) if isinstance(value, np.ndarray) else value
    if isinstance(scalar, np.generic):
        return cast(int | float, scalar.item())
    return scalar


def iter_csv_rows(result: EvaluationResult) -> Iterator[CsvRow]:
    """Yield the stable long-form CSV rows, with array fields reduced to their mean."""
    for sequence, scores in (*result.per_sequence.items(), ("COMBINED", result.combined)):
        for metric, fields in scores.items():
            for field, value in fields.items():
                yield sequence, metric, field, _to_csv_value(value)
