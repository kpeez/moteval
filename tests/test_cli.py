"""CLI tests: run in-process via `moteval.cli.main(argv)` -- the installed
console script isn't used because "toy" is registered only inside the test
suite (tests/conftest.py) and a subprocess entry point never sees it, and
because a nested `uv run` under `uv run pytest` deadlocks on uv's project lock.
"""

import csv
import json

import numpy as np
import pytest

from moteval import evaluate, load_dataset
from moteval.cli import main
from moteval.formats.mot_txt import write_mot
from moteval.results import EvaluationResult


@pytest.fixture
def toy_predictions(tmp_path):
    dataset = load_dataset("toy")
    pred_dir = tmp_path / "predictions"
    for sequence in dataset.sequences:
        write_mot(pred_dir / f"{sequence.name}.txt", list(sequence.tracks))
    return dataset, pred_dir


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


def test_toy_run_prints_sequence_and_combined_headlines(toy_predictions, capsys):
    _dataset, pred_dir = toy_predictions

    exit_code = main(["run", "--dataset", "toy", "--pred", str(pred_dir)])

    assert exit_code == 0
    lines = capsys.readouterr().out.splitlines()
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
    from moteval import CLEAR, HOTA, Count, Identity
    from moteval.results import iter_csv_rows, to_json_dict

    dataset, pred_dir = toy_predictions
    result = evaluate(dataset, pred_dir, [HOTA(), CLEAR(), Identity(), Count()])

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

    exit_code = main(
        [
            "run",
            "--dataset",
            "toy",
            "--pred",
            str(pred_dir),
            "--metrics",
            "hota,count",
            "--out-csv",
            str(out_csv),
        ]
    )

    assert exit_code == 0
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

    exit_code = main(
        ["run", "--dataset", "toy", "--pred", str(pred_dir), "--out-json", str(out_json)]
    )

    assert exit_code == 0
    with out_json.open() as file:
        exported = json.load(file)
    assert list(exported) == ["dataset", "split", "per_sequence", "combined"]
    assert exported["dataset"] == "toy"
    assert exported["split"] == "val"
    assert set(exported["per_sequence"]) == {"toy-0001", "toy-0002"}
    assert set(exported["per_sequence"]["toy-0001"]) == {"HOTA", "CLEAR", "Identity", "Count"}
    assert len(exported["combined"]["HOTA"]["HOTA"]) == 19


def test_json_export_round_trips_direct_evaluate_values(toy_predictions, tmp_path):
    from moteval import CLEAR, HOTA, Count, Identity

    dataset, pred_dir = toy_predictions
    out_json = tmp_path / "result.json"
    direct: EvaluationResult = evaluate(dataset, pred_dir, [HOTA(), CLEAR(), Identity(), Count()])

    exit_code = main(
        ["run", "--dataset", "toy", "--pred", str(pred_dir), "--out-json", str(out_json)]
    )

    assert exit_code == 0
    with out_json.open() as file:
        exported = json.load(file)
    assert exported["per_sequence"] == {
        sequence: _python_scores(scores) for sequence, scores in direct.per_sequence.items()
    }
    assert exported["combined"] == _python_scores(direct.combined)


def test_unknown_dataset_lists_registered_names(toy_predictions, capsys):
    _dataset, pred_dir = toy_predictions

    with pytest.raises(SystemExit):
        main(["run", "--dataset", "not-a-dataset", "--pred", str(pred_dir)])

    err = capsys.readouterr().err
    assert "unknown dataset 'not-a-dataset'" in err
    assert "registered:" in err
    assert "dancetrack" in err
    assert "toy" in err
    assert "Traceback" not in err


def test_unknown_metric_lists_available_names(toy_predictions, capsys):
    _dataset, pred_dir = toy_predictions

    with pytest.raises(SystemExit):
        main(["run", "--dataset", "toy", "--pred", str(pred_dir), "--metrics", "hota,not-a-metric"])

    err = capsys.readouterr().err
    assert "unknown metric(s): not-a-metric" in err
    assert "available: hota, clear, identity, count, track_map, jf" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--dataset", "toy", "--pred", "missing-predictions"], "prediction directory"),
        (
            ["--dataset", "dancetrack", "--pred", ".", "--gt", "missing-ground-truth"],
            "ground-truth root",
        ),
    ],
)
def test_missing_input_paths_are_actionable(args, message, capsys):
    with pytest.raises(SystemExit):
        main(["run", *args])

    err = capsys.readouterr().err
    assert message in err
    assert "not found or not a directory" in err
    assert "Traceback" not in err


# --------------------------------------------------------------- data command


_EXPECTED_BENCHMARKS = (
    "animaltrack",
    "bft",
    "chimpact",
    "dancetrack",
    "gmot40",
    "mots20",
    "panaf500",
    "sportsmot",
    "uavdt",
)


def test_data_list_reports_every_supported_benchmark(capsys):
    exit_code = main(["data", "list"])

    assert exit_code == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    assert tuple(row[0] for row in rows) == _EXPECTED_BENCHMARKS
    assert dict(rows)["chimpact"] == "unfetchable"


def test_data_status_uses_loader_layout_markers(tmp_path, capsys):
    (tmp_path / "animaltrack" / "gt_all").mkdir(parents=True)
    (tmp_path / "chimpact" / "ChimpACT_release_v1" / "labels").mkdir(parents=True)

    exit_code = main(["data", "status", "--root", str(tmp_path)])

    assert exit_code == 0
    rows = {
        parts[0]: parts[1]
        for line in capsys.readouterr().out.splitlines()[1:]
        if (parts := line.split())
    }
    assert rows["animaltrack"] == "present"
    assert rows["chimpact"] == "present"
    assert rows["dancetrack"] == "absent"


def test_data_root_option_overrides_environment(tmp_path, monkeypatch, capsys):
    environment_root = tmp_path / "environment"
    cli_root = tmp_path / "cli"
    (environment_root / "gmot40" / "track_label").mkdir(parents=True)
    monkeypatch.setenv("MOTEVAL_DATA_ROOT", str(environment_root))

    exit_code = main(["data", "--root", str(cli_root), "status"])

    assert exit_code == 0
    gmot40_row = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("gmot40")
    )
    assert "absent" in gmot40_row
    assert str(cli_root / "gmot40") in gmot40_row


def test_data_status_uses_environment_root(tmp_path, monkeypatch, capsys):
    (tmp_path / "sportsmot" / "val").mkdir(parents=True)
    monkeypatch.setenv("MOTEVAL_DATA_ROOT", str(tmp_path))

    exit_code = main(["data", "status"])

    assert exit_code == 0
    sportsmot_row = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("sportsmot")
    )
    assert "present" in sportsmot_row
    assert str(tmp_path / "sportsmot") in sportsmot_row


def test_unknown_download_lists_valid_benchmark_names(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["data", "download", "not-a-benchmark", "--root", str(tmp_path)])

    err = capsys.readouterr().err
    assert "unknown benchmark 'not-a-benchmark'" in err
    assert "valid names: " + ", ".join(_EXPECTED_BENCHMARKS) in err
    assert "Traceback" not in err


def test_unfetchable_download_gives_manual_acquisition_layout(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["data", "download", "chimpact", "--root", str(tmp_path)])

    err = capsys.readouterr().err
    assert "no stable programmatic artifact" in err
    assert "https://github.com/ShirleyMaxx/ChimpACT" in err
    assert "<root>/chimpact/ChimpACT_release_v1/labels/<clip>.json" in err
    assert "Traceback" not in err
