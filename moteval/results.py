"""Typed evaluation results: per-sequence and combined metric scores."""

from dataclasses import dataclass

from moteval.metrics.base import Scores

# metric name -> field -> value
MetricScores = dict[str, Scores]


@dataclass(frozen=True)
class EvaluationResult:
    """Scores for one evaluation run.

    ``per_sequence`` maps sequence name -> metric name -> field -> value;
    ``combined`` maps metric name -> field -> value across all sequences.
    """

    per_sequence: dict[str, MetricScores]
    combined: MetricScores
