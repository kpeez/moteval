"""Top-level evaluation: score a loaded `MOTDataset` against prediction files.

`evaluate(dataset, predictions, metrics)` reads ``<seq>.txt`` MOTChallenge (or
MOTS) predictions from a directory and returns typed results. Every sequence
needs a file; an empty file means no predictions. A missing file raises
``ValueError`` before any sequence is scored, and so does a metric whose
`Metric.geometry` (boxes or masks) does not match a sequence. It is the whole
public scoring seam: any `MOTDataset` — built-in benchmark, generic-layout load,
or hand-constructed — evaluates through it.

Each class in ``protocol.eval_classes`` is scored on its own: one `SequenceData`
view per sequence, every metric, then the sequences combined for that class. A
single-class run reports those scores as ``per_sequence`` and ``combined``. A
multi-class run reports them under ``per_class`` and also combines the classes
in the two ways TrackEval does: ``class_averaged`` and ``det_averaged``. Box
predictions give their class in column 8, which is read only for multi-class
protocols.

Results hold only each metric's declared ``fields``. The raw per-sequence and
combined state, which can carry private keys (TrackMAP's match arrays and
``_num_dt_*`` weights), stays inside `evaluate`, where the class combiners use it.

The whole run (reading, conversion, scoring, combining) happens in the Rust extension
(``rust/moteval/src/eval.rs``) with the GIL released, so `evaluate` takes the built-in
metrics only.
"""

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
