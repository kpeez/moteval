"""Type stubs for the Rust extension `moteval._moteval` (rust/moteval-py/src/lib.rs)."""

from collections.abc import Iterable
from os import PathLike
from typing import Any

from moteval.data.model import MOTDataset
from moteval.metrics.base import Metric
from moteval.results import EvaluationResult

def evaluate(
    dataset: MOTDataset[Any],
    predictions: str | PathLike[str],
    metrics: Iterable[Metric | str],
) -> dict[str, Any]: ...
def results_table(result: EvaluationResult) -> str: ...
def results_to_json(result: EvaluationResult, dataset: str, split: str) -> str: ...
def results_to_csv(result: EvaluationResult) -> str: ...
def cli_run(argv: list[str]) -> tuple[int, str, str]: ...
def main() -> int: ...
