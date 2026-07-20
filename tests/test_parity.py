"""Parity vs frozen TrackEval oracle numbers on the shared synthetic scenarios.

The JSON fixtures under ``tests/fixtures/`` hold the output of official TrackEval
(pinned commit, numpy>=2 alias patches only) on every scenario in
``tests/scenarios.py``. They are never hand-edited; regenerate them with
``scripts/regen_parity_fixtures.py``. Equality is exact — any divergence is a
parity regression, not a tolerance problem.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from moteval import CLEAR, HOTA, Count, Identity, JAndF, TrackMAP, evaluate
from tests.scenarios import (
    BOX_SCENARIOS,
    COMBINE_CLASSES_SCENARIO,
    TRACKMAP_SCENARIOS,
    build_box_dataset,
    build_mots_dataset,
    build_mots_scenarios,
    build_trackmap_sequence_data,
    predictions_dir,
    write_mot_scenario,
    write_mots_scenario,
)

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_BOX = json.loads((FIXTURES / "synthetic_box.json").read_text())
SYNTHETIC_MOTS = json.loads((FIXTURES / "synthetic_mots.json").read_text())
SYNTHETIC_TRACKMAP = json.loads((FIXTURES / "synthetic_trackmap.json").read_text())

BOX_METRICS = (HOTA, CLEAR, Identity, Count)
MOTS_METRICS = (HOTA, CLEAR, Identity, Count, JAndF)
MOTS_SCENARIOS = build_mots_scenarios()


def _assert_fields_equal(scores, frozen: dict, label: str) -> None:
    for field, frozen_value in frozen.items():
        assert field in scores, f"{label}.{field} missing from moteval output"
        ours = np.asarray(scores[field], dtype=float)
        oracle = np.asarray(frozen_value, dtype=float)
        assert np.array_equal(ours, oracle, equal_nan=True), (
            f"{label}.{field}: moteval={ours} oracle={oracle}"
        )


@pytest.mark.parametrize("scenario", BOX_SCENARIOS, ids=lambda s: s.name)
def test_box_parity(scenario, tmp_path):
    write_mot_scenario(tmp_path, scenario)
    dataset = build_box_dataset(tmp_path, scenario)
    result = evaluate(dataset, predictions_dir(tmp_path), [m() for m in BOX_METRICS])
    frozen = SYNTHETIC_BOX[scenario.name]
    for metric in BOX_METRICS:
        name = metric.__name__
        _assert_fields_equal(result.combined[name], frozen[name], f"{scenario.name}.{name}")


def test_hota_combine_classes_class_averaged(tmp_path):
    scenario = next(s for s in BOX_SCENARIOS if s.name == COMBINE_CLASSES_SCENARIO)
    write_mot_scenario(tmp_path, scenario)
    result = evaluate(build_box_dataset(tmp_path, scenario), predictions_dir(tmp_path), [HOTA()])
    all_res = {
        "class_a": result.per_sequence["CLASSA01"]["HOTA"],
        "class_b": result.per_sequence["CLASSB01"]["HOTA"],
    }
    class_avg = HOTA().combine_classes_class_averaged(all_res)
    frozen = SYNTHETIC_BOX["combine_classes_class_averaged"]["HOTA"]
    _assert_fields_equal(class_avg, frozen, "combine_classes_class_averaged.HOTA")

    # det_averaged is moteval's sole intentional divergence from upstream (upstream
    # bug); it is deliberately absent from the fixtures. Prove the combiners differ
    # so the class-averaged check above has teeth.
    det_avg = HOTA().combine_classes_det_averaged(all_res)
    assert not np.array_equal(class_avg["DetA"], det_avg["DetA"])


@pytest.mark.parametrize("scenario", MOTS_SCENARIOS, ids=lambda s: s.name)
def test_mots_parity(scenario, tmp_path):
    write_mots_scenario(tmp_path, scenario)
    dataset = build_mots_dataset(tmp_path, scenario)
    result = evaluate(dataset, predictions_dir(tmp_path), [m() for m in MOTS_METRICS])
    frozen = SYNTHETIC_MOTS[scenario.name]
    for metric in MOTS_METRICS:
        name = metric.__name__
        _assert_fields_equal(result.combined[name], frozen[name], f"{scenario.name}.{name}")


@pytest.mark.parametrize("name", sorted(TRACKMAP_SCENARIOS), ids=str)
def test_trackmap_parity(name):
    sequences = TRACKMAP_SCENARIOS[name]
    metric = TrackMAP()
    per_seq = {
        seq_name: metric.eval_sequence(
            build_trackmap_sequence_data(num_timesteps, gt_tracks, pred_tracks)
        )
        for seq_name, (num_timesteps, gt_tracks, pred_tracks) in sequences.items()
    }
    combined = metric.combine_sequences(per_seq)
    _assert_fields_equal(combined, SYNTHETIC_TRACKMAP[name]["TrackMAP"], f"{name}.TrackMAP")
