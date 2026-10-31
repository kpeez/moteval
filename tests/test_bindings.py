"""The Rust extension `moteval._moteval` against the Python code it is to replace.

Until the Rust loaders and metrics land, these cover what the extension can run on its
own: the CLI on arguments that stop before a dataset is loaded, the table and export
writers on results Python's `evaluate` computes, and `evaluate`'s errors before scoring.
"""

import contextlib
import io
import json
import sys
from dataclasses import replace

import pytest

from moteval import (
    CLEAR,
    HOTA,
    Count,
    Identity,
    JAndF,
    TrackMAP,
    _moteval,
    evaluate,
)
from moteval.cli import _table, _write_csv, main
from moteval.results import to_json_dict
from tests.conftest import (
    load_toy,
    load_two_class,
    write_perfect_predictions,
    write_two_class_predictions,
)


def _python_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = exc.code
    assert isinstance(code, int)
    return code, stdout.getvalue(), stderr.getvalue()


# The Rust CLI copies the text of CPython 3.14's argparse; older interpreters word and
# lay out some argparse messages differently. rust/moteval/tests/cli.rs checks every
# argparse case against CPython 3.14.8.
_ARGPARSE_3_14 = pytest.mark.skipif(
    sys.version_info[:2] != (3, 14), reason="the Rust CLI copies argparse 3.14's text"
)


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(argv, marks=_ARGPARSE_3_14)
        for argv in (
            [],
            ["-h"],
            ["run", "-h"],
            ["frob"],
            ["run"],
            ["run", "--pred"],
            ["run", "--pred", ".", "--format", "png"],
            ["run", "--pred", ".", "--bogus"],
        )
    ]
    + [
        # moteval's own messages, after the top-level usage line.
        ["run", "--pred", "missing-predictions"],
        ["run", "--pred", "."],
        ["run", "--pred", ".", "--metrics", "hota,not-a-metric"],
        ["run", "--dataset", "dancetrack", "--pred", ".", "--gt", "missing-ground-truth"],
    ],
)
def test_rust_cli_matches_python_cli(argv, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # argparse wraps usage to the terminal width; the Rust CLI uses argparse's default.
    monkeypatch.setenv("COLUMNS", "80")
    for name in ("FORCE_COLOR", "PYTHON_COLORS"):
        monkeypatch.delenv(name, raising=False)

    assert _moteval.cli_run(argv) == _python_cli(argv)


def _toy_result(tmp_path):
    dataset = load_toy()
    write_perfect_predictions(dataset, tmp_path / "toy")
    metrics = [HOTA(), CLEAR(), Identity(), Count(), TrackMAP()]
    return dataset, evaluate(dataset, tmp_path / "toy", metrics)


def _two_class_result(tmp_path):
    write_two_class_predictions(tmp_path / "two")
    dataset = load_two_class()
    metrics = [HOTA(), CLEAR(), Identity(), Count(), TrackMAP()]
    return dataset, evaluate(dataset, tmp_path / "two", metrics)


@pytest.mark.parametrize("scored", [_toy_result, _two_class_result])
def test_rust_writers_match_python_bytes(scored, tmp_path):
    dataset, result = scored(tmp_path)
    csv_path = tmp_path / "result.csv"
    _write_csv(csv_path, result)

    assert _moteval.results_table(result) == _table(result)
    assert _moteval.results_to_json(result, dataset.name, dataset.split) == json.dumps(
        to_json_dict(result, dataset=dataset.name, split=dataset.split)
    )
    assert _moteval.results_to_csv(result) == csv_path.read_bytes().decode()


def test_rust_evaluate_raises_python_errors_before_scoring(tmp_path):
    toy = load_toy()
    pred_dir = tmp_path / "predictions"
    write_perfect_predictions(toy, pred_dir)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "toy-0002.txt").write_text("")
    cases = [
        (toy, pred_dir, [HOTA(), Count(), HOTA()]),
        (toy, pred_dir, [HOTA(), JAndF()]),
        (replace(toy, protocol=replace(toy.protocol, eval_classes=())), pred_dir, [HOTA()]),
        (replace(toy, protocol=replace(toy.protocol, eval_classes=(1, 1))), pred_dir, [HOTA()]),
        (toy, partial, [Count()]),
        (toy, tmp_path / "missing" / ".", [Count()]),
    ]
    for dataset, predictions, metrics in cases:
        with pytest.raises(ValueError) as python_error:
            evaluate(dataset, predictions, metrics)
        with pytest.raises(ValueError) as rust_error:
            _moteval.evaluate(dataset, predictions, metrics)
        assert str(rust_error.value) == str(python_error.value)
