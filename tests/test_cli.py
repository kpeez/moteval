import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from moteval import CLEAR, HOTA, Count, Identity, evaluate, load_dataset
from moteval.formats.mot_txt import write_mot
from moteval.results import iter_csv_rows, to_json_dict

ROOT = Path(__file__).parents[1]
DEFAULT_METRICS = [HOTA(), CLEAR(), Identity(), Count()]


@pytest.fixture
def toy_predictions(tmp_path):
    dataset = load_dataset("toy")
    pred_dir = tmp_path / "predictions"
    for sequence in dataset.sequences:
        write_mot(pred_dir / f"{sequence.name}.txt", list(sequence.tracks))
    return dataset, pred_dir


def _run_cli(*args):
    # The installed console script is invoked directly: a nested `uv run` would
    # deadlock on uv's project lock while the outer `uv run pytest` holds it.
    entry_point = ROOT / ".venv" / "bin" / "moteval"
    return subprocess.run(
        [str(entry_point), "run", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _python_scores(scores):
    return {
        metric: {
            field: value.tolist()
            if isinstance(value, np.ndarray)
            else value.item()
            if isinstance(value, np.generic)
            else value
            for field, value in fields.items()
        }
        for metric, fields in scores.items()
    }


def test_toy_run_prints_sequence_and_combined_headlines(toy_predictions):
    _dataset, pred_dir = toy_predictions

    completed = _run_cli("--dataset", "toy", "--pred", pred_dir)

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0].split() == [
        "seq",
        "HOTA",
        "DetA",
        "AssA",
        "MOTA",
        "MOTP",
        "IDSW",
        "IDF1",
        "Dets",
        "GT_Dets",
    ]
    assert lines[1].split() == [
        "toy-0001",
        "100",
        "100",
        "100",
        "100",
        "100",
        "0",
        "100",
        "10",
        "10",
    ]
    assert lines[2].split() == [
        "toy-0002",
        "100",
        "100",
        "100",
        "100",
        "100",
        "0",
        "100",
        "10",
        "10",
    ]
    assert lines[3].split() == [
        "COMBINED",
        "100",
        "100",
        "100",
        "100",
        "100",
        "0",
        "100",
        "20",
        "20",
    ]


def test_result_serializers_follow_stable_schemas(toy_predictions):
    dataset, pred_dir = toy_predictions
    result = evaluate(dataset, pred_dir, DEFAULT_METRICS)

    json_result = to_json_dict(result, dataset=dataset.name, split=dataset.split)
    assert list(json_result) == ["dataset", "split", "per_sequence", "combined"]
    assert json_result["dataset"] == "toy"
    assert json_result["split"] == "val"
    hota = json_result["per_sequence"]["toy-0001"]["HOTA"]["HOTA"]
    assert isinstance(hota, list)
    assert len(hota) == 19
    assert hota == [1.0] * 19

    rows = list(iter_csv_rows(result))
    assert rows[0] == ("toy-0001", "HOTA", "HOTA", 1.0)
    assert ("toy-0001", "HOTA", "HOTA_TP", 10.0) in rows
    assert ("COMBINED", "Count", "GT_Dets", 20.0) in rows
    assert all(isinstance(value, (int, float)) for _seq, _metric, _field, value in rows)


def test_cli_writes_csv_with_stable_schema(toy_predictions, tmp_path):
    _dataset, pred_dir = toy_predictions
    out_csv = tmp_path / "exports" / "result.csv"

    completed = _run_cli(
        "--dataset",
        "toy",
        "--pred",
        pred_dir,
        "--metrics",
        "hota,count",
        "--out-csv",
        out_csv,
    )

    assert completed.returncode == 0, completed.stderr
    with out_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == ["seq", "metric", "field", "value"]
    assert {row["seq"] for row in rows} == {"toy-0001", "toy-0002", "COMBINED"}
    assert {row["metric"] for row in rows} == {"HOTA", "Count"}
    assert (
        next(
            row
            for row in rows
            if row["seq"] == "toy-0001" and row["metric"] == "HOTA" and row["field"] == "HOTA"
        )["value"]
        == "1.0"
    )


def test_cli_writes_json_with_stable_schema(toy_predictions, tmp_path):
    _dataset, pred_dir = toy_predictions
    out_json = tmp_path / "exports" / "result.json"

    completed = _run_cli(
        "--dataset",
        "toy",
        "--pred",
        pred_dir,
        "--out-json",
        out_json,
    )

    assert completed.returncode == 0, completed.stderr
    with out_json.open() as file:
        exported = json.load(file)
    assert list(exported) == ["dataset", "split", "per_sequence", "combined"]
    assert exported["dataset"] == "toy"
    assert exported["split"] == "val"
    assert set(exported["per_sequence"]) == {"toy-0001", "toy-0002"}
    assert set(exported["per_sequence"]["toy-0001"]) == {
        "HOTA",
        "CLEAR",
        "Identity",
        "Count",
    }
    assert len(exported["combined"]["HOTA"]["HOTA"]) == 19


def test_json_export_round_trips_direct_evaluate_values(toy_predictions, tmp_path):
    dataset, pred_dir = toy_predictions
    out_json = tmp_path / "result.json"
    direct = evaluate(dataset, pred_dir, DEFAULT_METRICS)

    completed = _run_cli(
        "--dataset",
        "toy",
        "--pred",
        pred_dir,
        "--out-json",
        out_json,
    )

    assert completed.returncode == 0, completed.stderr
    with out_json.open() as file:
        exported = json.load(file)
    assert exported["per_sequence"] == {
        sequence: _python_scores(scores) for sequence, scores in direct.per_sequence.items()
    }
    assert exported["combined"] == _python_scores(direct.combined)


def test_unknown_dataset_lists_registered_names(toy_predictions):
    _dataset, pred_dir = toy_predictions

    completed = _run_cli("--dataset", "not-a-dataset", "--pred", pred_dir)

    assert completed.returncode != 0
    assert "unknown dataset 'not-a-dataset'" in completed.stderr
    assert "registered:" in completed.stderr
    assert "dancetrack" in completed.stderr
    assert "toy" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_unknown_metric_lists_available_names(toy_predictions):
    _dataset, pred_dir = toy_predictions

    completed = _run_cli("--dataset", "toy", "--pred", pred_dir, "--metrics", "hota,not-a-metric")

    assert completed.returncode != 0
    assert "unknown metric(s): not-a-metric" in completed.stderr
    assert "available: hota, clear, identity, count, track_map, jf" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--dataset", "toy", "--pred", "missing-predictions"), "prediction directory"),
        (
            ("--dataset", "dancetrack", "--pred", ".", "--gt", "missing-ground-truth"),
            "ground-truth root",
        ),
    ],
)
def test_missing_input_paths_are_actionable(args, message):
    completed = _run_cli(*args)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert "not found or not a directory" in completed.stderr
    assert "Traceback" not in completed.stderr
