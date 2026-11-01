from collections.abc import Sequence
from pathlib import Path

from moteval import _moteval
from moteval._rust import call
from moteval.data.model import MOTDataset
from moteval.metrics import CLEAR, HOTA, Count, Identity, JAndF, TrackMAP
from moteval.metrics.base import Metric
from moteval.results import ClassResult, EvaluationResult

_BUILT_IN = (HOTA, CLEAR, Identity, Count, TrackMAP, JAndF)


def evaluate(
    dataset: MOTDataset,
    predictions: str | Path,
    metrics: Sequence[Metric],
) -> EvaluationResult:
    """Score ``dataset`` against ``predictions/<seq>.txt``; an empty file means no predictions.

    Raises `ValueError` before scoring on a missing file, a metric that is not built in, or
    a metric whose `Metric.geometry` does not match a sequence.
    """
    for metric in metrics:
        if type(metric) not in _BUILT_IN:
            built_in = ", ".join(cls.__name__ for cls in _BUILT_IN)
            raise ValueError(
                f"evaluate runs the built-in metrics only ({built_in}); got {type(metric).__name__}"
            )
    names = [type(metric).__name__ for metric in metrics]
    fields = call(_moteval.evaluate, dataset, predictions, names)
    fields["per_class"] = {
        cls_id: ClassResult(**result) for cls_id, result in fields["per_class"].items()
    }
    return EvaluationResult(**fields)
