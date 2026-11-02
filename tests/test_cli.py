import csv
import io
import json

import numpy as np
import pytest

from moteval import _moteval, evaluate, load_motchallenge
from moteval.cli import main
from moteval.formats import write_mot
from moteval.results import EvaluationResult, to_csv, to_json_dict
from tests.conftest import (
    load_two_class,
    write_flawed_predictions,
    write_perfect_predictions,
    write_toy_layout,
    write_two_class_predictions,
)


@pytest.fixture
def toy_gt(tmp_path):
    root = tmp_path / "toy"
    write_toy_layout(root)
    return root


@pytest.fixture
def toy_predictions(tmp_path, toy_gt):
    dataset = load_motchallenge(toy_gt, "val")
    pred_dir = tmp_path / "predictions"
    write_perfect_predictions(dataset, pred_dir)
    return dataset, pred_dir


def _toy_args(toy_gt) -> list[str]:
    return ["--gt", str(toy_gt), "--split", "val"]


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


def test_toy_run_prints_sequence_and_combined_headlines(tmp_path, toy_gt, capsys):
    pred_dir = tmp_path / "predictions"
    write_flawed_predictions(load_motchallenge(toy_gt, "val"), pred_dir)

    exit_code = main(["run", *_toy_args(toy_gt), "--pred", str(pred_dir)])

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
    ratios = ["59.161", "75", "46.667", "50", "100"]
    assert lines[1].split() == ["toy-0001", *ratios, "2", "47.619", "11", "10"]
    assert lines[2].split() == ["toy-0002", *ratios, "2", "47.619", "11", "10"]
    assert lines[3].split() == ["COMBINED", *ratios, "4", "47.619", "22", "20"]


def test_result_serializers_follow_stable_schemas(toy_predictions):
    from moteval import CLEAR, HOTA, Count, Identity

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

    header, *rows = csv.reader(io.StringIO(to_csv(result), newline=""))
    assert header == ["seq", "metric", "field", "value"]
    assert rows[0] == ["toy-0001", "HOTA", "HOTA", "1.0"]
    assert ["toy-0001", "HOTA", "HOTA_TP", "10.0"] in rows
    assert ["COMBINED", "Count", "GT_Dets", "20.0"] in rows
    assert all(isinstance(float(row[-1]), float) for row in rows)


def test_cli_writes_csv_with_stable_schema(toy_predictions, toy_gt, tmp_path):
    _dataset, pred_dir = toy_predictions
    out_csv = tmp_path / "exports" / "result.csv"

    exit_code = main(
        [
            "run",
            *_toy_args(toy_gt),
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


def test_cli_writes_json_with_stable_schema(toy_predictions, toy_gt, tmp_path):
    _dataset, pred_dir = toy_predictions
    out_json = tmp_path / "exports" / "result.json"

    exit_code = main(
        ["run", *_toy_args(toy_gt), "--pred", str(pred_dir), "--out-json", str(out_json)]
    )

    assert exit_code == 0
    with out_json.open() as file:
        exported = json.load(file)
    assert list(exported) == ["dataset", "split", "per_sequence", "combined"]
    assert exported["dataset"] == "toy"
    assert exported["split"] == "val"


def test_json_export_round_trips_direct_evaluate_values(toy_predictions, toy_gt, tmp_path):
    from moteval import CLEAR, HOTA, Count, Identity

    dataset, pred_dir = toy_predictions
    out_json = tmp_path / "result.json"
    direct: EvaluationResult = evaluate(dataset, pred_dir, [HOTA(), CLEAR(), Identity(), Count()])

    exit_code = main(
        ["run", *_toy_args(toy_gt), "--pred", str(pred_dir), "--out-json", str(out_json)]
    )

    assert exit_code == 0
    with out_json.open() as file:
        exported = json.load(file)
    assert exported["per_sequence"] == {
        sequence: _python_scores(scores) for sequence, scores in direct.per_sequence.items()
    }
    assert exported["combined"] == _python_scores(direct.combined)


def test_single_class_outputs_keep_their_exact_format(toy_predictions, toy_gt, tmp_path, capsys):
    _dataset, pred_dir = toy_predictions
    out_csv = tmp_path / "result.csv"
    out_json = tmp_path / "result.json"

    exit_code = main(
        [
            "run",
            *_toy_args(toy_gt),
            "--pred",
            str(pred_dir),
            "--metrics",
            "count",
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "seq       Dets  GT_Dets\n"
        "toy-0001    10       10\n"
        "toy-0002    10       10\n"
        "COMBINED    20       20\n"
    )
    csv_rows = ["seq,metric,field,value"]
    for seq, (dets, ids) in {"toy-0001": (10, 2), "toy-0002": (10, 2), "COMBINED": (20, 4)}.items():
        csv_rows += [
            f"{seq},Count,Dets,{dets}.0",
            f"{seq},Count,GT_Dets,{dets}.0",
            f"{seq},Count,IDs,{ids}.0",
            f"{seq},Count,GT_IDs,{ids}.0",
        ]
    assert out_csv.read_bytes() == ("\r\n".join(csv_rows) + "\r\n").encode()
    per_sequence = '{"Count": {"Dets": 10.0, "GT_Dets": 10.0, "IDs": 2.0, "GT_IDs": 2.0}}'
    assert out_json.read_text() == (
        '{"dataset": "toy", "split": "val", "per_sequence": '
        f'{{"toy-0001": {per_sequence}, "toy-0002": {per_sequence}}}, '
        '"combined": {"Count": {"Dets": 20.0, "GT_Dets": 20.0, "IDs": 4.0, "GT_IDs": 4.0}}}'
    )


def test_multi_class_outputs_add_a_class_column_and_both_combinations(tmp_path):
    from moteval import HOTA, Count

    pred_dir = tmp_path / "predictions"
    write_two_class_predictions(pred_dir)
    dataset = load_two_class()
    result = evaluate(dataset, pred_dir, [HOTA(), Count()])
    out_csv = tmp_path / "result.csv"
    out_json = tmp_path / "result.json"
    out_csv.write_text(to_csv(result), newline="")
    out_json.write_text(_moteval.results_to_json(result, dataset.name, dataset.split))

    raw_lines = _moteval.results_table(result).splitlines()
    lines = [line.split() for line in raw_lines]
    seq_start = raw_lines[0].index("seq")
    assert all(
        raw[seq_start:].startswith(row[1]) for raw, row in zip(raw_lines, lines, strict=True)
    )
    seq = "two-class-0001"
    assert lines == [
        ["class", "seq", "HOTA", "DetA", "AssA", "Dets", "GT_Dets"],
        ["1", seq, "81.65", "66.667", "100", "3", "2"],
        ["1", "COMBINED", "81.65", "66.667", "100", "3", "2"],
        ["2", seq, "70.711", "50", "100", "2", "4"],
        ["2", "COMBINED", "70.711", "50", "100", "2", "4"],
        ["class_averaged", "COMBINED", "76.18", "58.333", "100", "5", "6"],
        ["det_averaged", "COMBINED", "75.593", "57.143", "100", "5", "6"],
    ]

    with out_csv.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)
    assert header == ["class", "seq", "metric", "field", "value"]
    assert ["1", seq, "Count", "Dets", "3.0"] in rows
    assert ["2", "COMBINED", "Count", "GT_Dets", "4.0"] in rows
    assert ["class_averaged", "COMBINED", "Count", "GT_Dets", "6.0"] in rows
    assert ["det_averaged", "COMBINED", "Count", "Dets", "5.0"] in rows
    assert {(row[0], row[1]) for row in rows} == {
        ("1", seq),
        ("1", "COMBINED"),
        ("2", seq),
        ("2", "COMBINED"),
        ("class_averaged", "COMBINED"),
        ("det_averaged", "COMBINED"),
    }

    exported = json.loads(out_json.read_text())
    assert list(exported) == [
        "dataset",
        "split",
        "per_sequence",
        "combined",
        "per_class",
        "class_averaged",
        "det_averaged",
    ]
    assert exported["per_sequence"] == {}
    assert exported["combined"] == {}
    assert list(exported["per_class"]) == ["1", "2"]
    assert exported["per_class"]["2"]["per_sequence"][seq]["Count"]["GT_Dets"] == 4.0
    assert exported["per_class"]["1"]["combined"]["Count"]["Dets"] == 3.0
    assert exported["class_averaged"]["Count"]["GT_Dets"] == 6.0
    assert exported["det_averaged"]["HOTA"]["DetA"] == pytest.approx([4 / 7] * 19)


def test_unknown_dataset_is_reported_without_traceback(toy_predictions, capsys):
    _dataset, pred_dir = toy_predictions

    with pytest.raises(SystemExit):
        main(["run", "--dataset", "not-a-dataset", "--pred", str(pred_dir)])

    err = capsys.readouterr().err
    assert "unknown dataset 'not-a-dataset'" in err
    assert "Traceback" not in err


def test_unknown_metric_lists_available_names(toy_predictions, toy_gt, capsys):
    _dataset, pred_dir = toy_predictions

    with pytest.raises(SystemExit):
        main(["run", *_toy_args(toy_gt), "--pred", str(pred_dir), "--metrics", "hota,not-a-metric"])

    err = capsys.readouterr().err
    assert "unknown metric(s): not-a-metric" in err
    assert "available: hota, clear, identity, count, track_map, jf" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--dataset", "dancetrack", "--pred", "missing-predictions"], "prediction directory"),
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


def test_missing_prediction_file_is_reported_without_traceback(tmp_path, toy_gt, capsys):
    pred_dir = tmp_path / "predictions"
    write_mot(pred_dir / "toy-0001.txt", [])

    with pytest.raises(SystemExit) as exc:
        main(["run", *_toy_args(toy_gt), "--pred", str(pred_dir)])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"prediction directory {pred_dir} has no <seq>.txt file" in err
    assert "toy-0002" in err
    assert "Traceback" not in err


def test_metric_on_the_other_geometry_is_reported_without_traceback(
    toy_predictions, toy_gt, capsys
):
    _, pred_dir = toy_predictions

    with pytest.raises(SystemExit) as exc:
        main(["run", *_toy_args(toy_gt), "--pred", str(pred_dir), "--metrics", "jf"])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "JAndF reads masks, but sequence 'toy-0001' of dataset 'toy' holds boxes" in err
    assert "Traceback" not in err


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
    perfect = ["100", "100", "100", "100", "100", "0", "100", "2", "2"]
    assert rows == [["SEQ01", *perfect], ["COMBINED", *perfect]]


def test_run_without_dataset_requires_gt(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--pred", "."])

    err = capsys.readouterr().err
    assert "--gt is required when --dataset is omitted" in err
    assert "Traceback" not in err
