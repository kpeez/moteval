"""CLI tests: run in-process via `moteval.cli.main(argv)` -- the installed
console script isn't used because "toy" is registered in `BENCHMARKS` only
inside the test process (the `toy_benchmark` fixture in tests/conftest.py) and
a subprocess entry point never sees it, and because a nested `uv run` under
`uv run pytest` deadlocks on uv's project lock.
"""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from moteval import evaluate
from moteval.cli import main
from moteval.formats import Track, write_mot
from moteval.results import EvaluationResult
from tests.conftest import load_toy, write_perfect_predictions


@pytest.fixture
def toy_predictions(tmp_path, toy_benchmark):
    dataset = load_toy()
    pred_dir = tmp_path / "predictions"
    write_perfect_predictions(dataset, pred_dir)
    return dataset, pred_dir


def _write_flawed_predictions(dataset, pred_dir):
    """Per sequence: GT 1 -> ids 11 (frames 1-2), 12 (frames 3-5); GT 2 -> ids 13
    (frames 1-2), 14 (frames 3-4), frame 5 dropped; id 15 is a far-away box in
    frames 1-2. Every kept box copies its GT box, so each IoU is exactly 1 or 0."""
    pred_ids = {(1, 1): 11, (1, 2): 11, (1, 3): 12, (1, 4): 12, (1, 5): 12}
    pred_ids |= {(2, 1): 13, (2, 2): 13, (2, 3): 14, (2, 4): 14}
    for sequence in dataset.sequences:
        preds = [
            replace(t, track_id=pred_ids[t.track_id, t.frame])
            for t in sequence.tracks
            if (t.track_id, t.frame) in pred_ids
        ]
        preds += [Track(frame=f, track_id=15, x=500, y=500, w=10, h=10, conf=1.0) for f in (1, 2)]
        write_mot(pred_dir / f"{sequence.name}.txt", preds)


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


def test_toy_run_prints_sequence_and_combined_headlines(tmp_path, toy_benchmark, capsys):
    pred_dir = tmp_path / "predictions"
    _write_flawed_predictions(load_toy(), pred_dir)

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
    # Hand derivation per sequence (10 GT dets, 11 predictions, IoU 1 or 0, so every
    # HOTA alpha agrees): TP = 9, FN = 1 (GT 2 frame 5), FP = 2 (id 15).
    # DetA = 9 / (9 + 1 + 2) = 3/4. AssA = sum(TPA * TPA / (gt + pred - TPA)) / TP over
    # (GT, pred) pairs (1,11) 2/5, (1,12) 3/5, (2,13) 2/5, (2,14) 2/5 = (4+9+4+4)/45 = 7/15.
    # HOTA = sqrt(3/4 * 7/15) = sqrt(0.35) = 0.59161.
    # IDSW = 2 (GT 1 and GT 2 change id at frame 3). MOTA = (9 - 2 - 2) / 10 = 1/2.
    # MOTP = 1 (every match has IoU 1). IDF1: best id pairs (1,12) 3 + (2,13) 2 give
    # IDTP = 5, so IDF1 = 2 * 5 / (10 + 11) = 10/21 = 0.47619.
    # Both sequences repeat the pattern, so COMBINED keeps every ratio and doubles counts.
    ratios = ["59.161", "75", "46.667", "50", "100"]
    assert lines[1].split() == ["toy-0001", *ratios, "2", "47.619", "11", "10"]
    assert lines[2].split() == ["toy-0002", *ratios, "2", "47.619", "11", "10"]
    assert lines[3].split() == ["COMBINED", *ratios, "4", "47.619", "22", "20"]


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


def test_unknown_dataset_is_reported_without_traceback(toy_predictions, capsys):
    _dataset, pred_dir = toy_predictions

    with pytest.raises(SystemExit):
        main(["run", "--dataset", "not-a-dataset", "--pred", str(pred_dir)])

    err = capsys.readouterr().err
    assert "unknown dataset 'not-a-dataset'" in err
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


# ------------------------------------------------------- custom data (no --dataset)


def test_run_without_dataset_loads_motchallenge_layout(tmp_path, capsys):
    seq_dir = tmp_path / "gt" / "train" / "SEQ01"
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "gt" / "gt.txt").write_text("1,1,10,10,20,20,1\n2,1,12,10,20,20,1\n")
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nseqLength=2\n")
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    (pred_dir / "SEQ01.txt").write_text("1,7,10,10,20,20,1\n2,7,12,10,20,20,1\n")

    exit_code = main(["run", "--gt", str(tmp_path / "gt"), "--pred", str(pred_dir)])

    assert exit_code == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    # predictions repeat the GT boxes under another id: every ratio is perfect.
    perfect = ["100", "100", "100", "100", "100", "0", "100", "2", "2"]
    assert rows == [["SEQ01", *perfect], ["COMBINED", *perfect]]


def test_run_without_dataset_requires_gt(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--pred", "."])

    err = capsys.readouterr().err
    assert "--gt is required when --dataset is omitted" in err
    assert "Traceback" not in err
