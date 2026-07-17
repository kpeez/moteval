"""Test-side helper that runs the vendored TrackEval oracle on a MOTChallenge-format
GT + predictions directory pair and returns its raw metric dicts.

Parity tests use this to assert field-by-field equality against moteval without each
reinventing oracle invocation. See VENDORED.md for the vendored oracle's provenance.
"""

import sys
from pathlib import Path

_ORACLE_DIR = Path(__file__).resolve().parent
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

from _trackeval.datasets import MotChallenge2DBox  # noqa: E402
from _trackeval.eval import Evaluator  # noqa: E402
from _trackeval.metrics import CLEAR, HOTA, Count, Identity  # noqa: E402

_METRICS = {"HOTA": HOTA, "CLEAR": CLEAR, "Identity": Identity, "Count": Count}


def run_mot_challenge(
    gt_folder: str | Path,
    trackers_folder: str | Path,
    seq_lengths: dict[str, int],
    *,
    tracker: str = "oracle",
    benchmark: str = "MOT17",
    split: str = "train",
    do_preproc: bool = True,
    metrics: tuple[str, ...] = ("HOTA", "CLEAR", "Identity", "Count"),
) -> dict[str, dict]:
    """Evaluate a MOTChallenge-format sequence pair through the TrackEval oracle.

    Expected on-disk layout (SKIP_SPLIT_FOL, so no BENCHMARK-SPLIT middle folder):
        {gt_folder}/{seq}/gt/gt.txt
        {trackers_folder}/{tracker}/data/{seq}.txt

    Returns the ``COMBINED_SEQ`` pedestrian results as ``{metric_name: {field: value}}``,
    e.g. ``result["CLEAR"]["MOTA"]`` and ``result["HOTA"]["HOTA"]`` (an array over alphas).
    """
    eval_config = {
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    }
    dataset_config = {
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(trackers_folder),
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": split,
        "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": do_preproc,
        "PRINT_CONFIG": False,
        "TRACKERS_TO_EVAL": [tracker],
        "SEQ_INFO": dict(seq_lengths),
    }
    metrics_list = [_METRICS[name]({"PRINT_CONFIG": False}) for name in metrics if name != "Count"]

    evaluator = Evaluator(eval_config)
    dataset = MotChallenge2DBox(dataset_config)
    output_res, _ = evaluator.evaluate([dataset], metrics_list)

    combined = output_res["MotChallenge2DBox"][tracker]["COMBINED_SEQ"]["pedestrian"]
    return {name: combined[name] for name in metrics}
