import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from moteval.benchmarks import default_data_root

_SCRIPT = Path(__file__).parents[1] / "scripts" / "reference_results.py"
_spec = importlib.util.spec_from_file_location("reference_results", _SCRIPT)
assert _spec is not None and _spec.loader is not None
reference_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference_results)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reference_benchmarks.json"
REFERENCE = json.loads(FIXTURE_PATH.read_text()) if FIXTURE_PATH.is_file() else {}


@pytest.mark.real_data
@pytest.mark.parametrize(
    ("benchmark", "split"),
    reference_results.REFERENCE_CASES,
    ids=[reference_results.case_key(b, s) for b, s in reference_results.REFERENCE_CASES],
)
def test_combined_results_equal_the_reference(benchmark: str, split: str) -> None:
    if not REFERENCE:
        pytest.skip(
            "SKIPPING REAL-DATA REFERENCE GATE: tests/fixtures/reference_benchmarks.json missing "
            "- regenerate it with `scripts/reference_results.py --reference-fixture`"
        )
    root = default_data_root()
    if not (root / benchmark).is_dir():
        pytest.skip(
            f"SKIPPING REAL-DATA REFERENCE GATE: {benchmark} not found under {root} - fetch it "
            f"with `scripts/download_benchmarks.py download {benchmark}` "
            "(chimpact: place the release by hand, see docs/DATASETS.md)"
        )

    ours = reference_results.evaluate_case(benchmark, split, root)["combined"]
    frozen = REFERENCE[reference_results.case_key(benchmark, split)]

    assert ours.keys() == frozen.keys()
    for metric, fields in frozen.items():
        assert ours[metric].keys() == fields.keys(), metric
        for field, expected in fields.items():
            assert np.array_equal(
                np.asarray(ours[metric][field], dtype=float),
                np.asarray(expected, dtype=float),
                equal_nan=True,
            ), f"{metric}.{field}: {ours[metric][field]} != {expected}"
