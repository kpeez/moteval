"""Real-data parity gate vs frozen TrackEval numbers.

DanceTrack val, SportsMOT val, and one MOTS20 mask sequence, scored against
seeded perturbed predictions (``tests/perturb.py``). The expected numbers in
``tests/fixtures/real_data.json`` come from official TrackEval run on the same
data with the same seeds — regenerate with
``scripts/regen_parity_fixtures.py --real-data``.

Each test skips loudly when its dataset is absent from ``data/benchmarks/``,
naming the ``scripts/download_benchmarks.py download`` script that fetches it. A fully-skipped
run does not count as the gate passing.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from moteval import CLEAR, HOTA, Count, Identity, JAndF, evaluate
from tests.perturb import perturb_box_tracks
from tests.scenarios import (
    DATA_ROOT,
    predictions_dir,
    prepare_dancetrack_val,
    prepare_mots20_sequence,
    prepare_sportsmot_val,
)

pytestmark = pytest.mark.real_data

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "real_data.json"
REAL_DATA = json.loads(FIXTURE_PATH.read_text()) if FIXTURE_PATH.is_file() else {}
BOX_METRICS = (HOTA, CLEAR, Identity, Count)


def _require(split_root: Path, benchmark: str, expected_sequences: int) -> None:
    if not REAL_DATA:
        pytest.skip(
            "SKIPPING REAL-DATA PARITY GATE: tests/fixtures/real_data.json missing — "
            "regenerate with `scripts/regen_parity_fixtures.py --real-data`"
        )
    if not split_root.is_dir() or not any(split_root.iterdir()):
        pytest.skip(
            f"SKIPPING REAL-DATA PARITY GATE: {benchmark} not found under {split_root} — "
            f"fetch it with `scripts/download_benchmarks.py download {benchmark}`"
        )
    # A split that is still syncing (e.g. cloud-storage hydration in progress)
    # skips loudly rather than evaluating a partial dataset.
    seq_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
    incomplete = [
        p.name
        for p in seq_dirs
        if not (p / "gt" / "gt.txt").is_file() or not (p / "seqinfo.ini").is_file()
    ]
    if len(seq_dirs) < expected_sequences or incomplete:
        pytest.skip(
            f"SKIPPING REAL-DATA PARITY GATE: {benchmark} under {split_root} looks incomplete "
            f"({len(seq_dirs)}/{expected_sequences} sequence dirs; missing gt/seqinfo: "
            f"{incomplete[:5]}) — still downloading? Re-run once "
            f"`scripts/download_benchmarks.py download {benchmark}` has finished."
        )


def _assert_metrics_equal(ours: dict, frozen: dict, metrics: tuple[type, ...]) -> None:
    for metric in metrics:
        name = metric.__name__
        for field, frozen_value in frozen[name].items():
            if field not in ours[name]:
                continue
            assert np.array_equal(
                np.asarray(ours[name][field], dtype=float),
                np.asarray(frozen_value, dtype=float),
                equal_nan=True,
            ), f"{name}.{field}: {ours[name][field]} != {frozen_value}"


def test_real_data_parity_dancetrack_val(tmp_path):
    _require(DATA_ROOT / "dancetrack" / "val", "dancetrack", expected_sequences=25)
    dataset, _, _ = prepare_dancetrack_val(tmp_path)
    assert len(dataset.sequences) > 0
    ours = evaluate(dataset, predictions_dir(tmp_path), [m() for m in BOX_METRICS]).combined
    _assert_metrics_equal(ours, REAL_DATA["dancetrack_val"], BOX_METRICS)


def test_real_data_parity_sportsmot_val(tmp_path):
    _require(DATA_ROOT / "sportsmot" / "val", "sportsmot", expected_sequences=45)
    dataset, _, _ = prepare_sportsmot_val(tmp_path)
    assert len(dataset.sequences) > 0
    ours = evaluate(dataset, predictions_dir(tmp_path), [m() for m in BOX_METRICS]).combined
    _assert_metrics_equal(ours, REAL_DATA["sportsmot_val"], BOX_METRICS)


def test_real_data_parity_mots20_mask_sequence(tmp_path):
    _require(DATA_ROOT / "mots20" / "train", "mots20", expected_sequences=4)
    dataset, _, _ = prepare_mots20_sequence(tmp_path)
    metrics = [HOTA(), CLEAR(), Identity(), Count(), JAndF()]
    ours = evaluate(dataset, predictions_dir(tmp_path), metrics).combined
    _assert_metrics_equal(ours, REAL_DATA["mots20"], (*BOX_METRICS, JAndF))


def test_perturbation_is_seeded_and_independently_numbered():
    from moteval.data.model import FrameConvention
    from moteval.formats import Track

    convention = FrameConvention(name="1-indexed", first_frame=1)
    gt = tuple(
        Track(frame=f, track_id=tid, x=100.0 * tid, y=50.0, w=20.0, h=40.0, conf=1.0)
        for f in range(1, 11)
        for tid in (3, 8)
    )
    a = perturb_box_tracks(gt, 10, convention, seed=7)
    b = perturb_box_tracks(gt, 10, convention, seed=7)
    c = perturb_box_tracks(gt, 10, convention, seed=8)
    assert a == b  # reproducible
    assert a != c  # seed actually matters
    gt_ids = {t.track_id for t in gt}
    assert not gt_ids & {t.track_id for t in a}  # independently numbered
