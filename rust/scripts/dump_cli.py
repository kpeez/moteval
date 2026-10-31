"""Dump what the Python CLI prints and returns for argument lists that stop before loading.

Writes ``rust/testdata/cli.json``, which ``rust/moteval/tests/cli.rs`` replays against the
Rust ``moteval`` binary byte for byte (stdout, stderr, exit code). Run with the reference
interpreter from the repository root:

    uv run python rust/scripts/dump_cli.py

Each case runs ``moteval.cli.main(argv)`` in-process, with ``COLUMNS=80`` (argparse wraps
usage and help to the terminal width) and the working directory set to a fixture tree:
``pred/`` and ``gt/`` (empty directories) and ``file.txt``. Every case must stop before
a dataset is loaded; the script fails if one calls a loader. The cases cover argparse's
parsing and errors, ``--metrics`` resolution, the path checks, and the ``tests/test_cli.py``
cases that need no dataset or metric.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moteval import cli

OUT = Path(__file__).resolve().parents[1] / "testdata" / "cli.json"
FIXTURE = {"dirs": ["pred", "gt"], "files": ["file.txt"]}

RUN = ["run", "--pred", "pred"]
CASES = [
    # help and top-level parsing
    [], ["-h"], ["--help"], ["--he"], ["--h"], ["-hx"], ["-hh"], ["-h=x"], ["--help=x"],
    ["--help="], ["-h", "run"], ["--version"], ["frob"], ["rn"], ["RUN"], ["-"], [""],
    ["--"], ["--", "run"], ["-x"], ["-x", "run"], ["-x", "frob"], ["--=x"], ["---"],
    ["-1"], ["-1", "run"], ["- x"], ["-x y"],
    # run: help
    ["run", "-h"], ["run", "--help"], ["run", "--hel"], ["run", "-hx"], ["run", "-h=x"],
    ["run", "--help=x"], ["run", "--pred", "pred", "-h"], ["run", "--format", "png", "-h"],
    ["run", "-h", "--format", "png"], ["run", "--bogus", "-h"], ["run", "--pred", "-h"],
    # run: argparse errors
    ["run"], ["run", "--pred"], ["run", "--pred", "-x"], ["run", "--pred", "--split"],
    ["run", "--pred", "--", "pred"], ["run", "--dataset", "toy"],
    [*RUN, "--format", "png"], [*RUN, "--format", "MOT"], [*RUN, "--format", ""],
    [*RUN, "--format=mots", "--format", "x"], [*RUN, "extra"], [*RUN, "extra", "more"],
    [*RUN, "--out", "a"], [*RUN, "--out=a"], [*RUN, "--o", "a"], [*RUN, "--bogus"],
    [*RUN, "--bogus", "val"], [*RUN, "--bogus=val"], [*RUN, "--", "x"], [*RUN, "--"],
    [*RUN, "-"], [*RUN, "-p", "x"], [*RUN, "-pred"], [*RUN, "--gt"], [*RUN, "--out-json"],
    [*RUN, "--=x"], [*RUN, "--split", "-x"], [*RUN, "--split", "-1"], ["-x", *RUN, "--y"],
    ["run", "--=x", "--pred", "pred"], ["run", "--pred=pred", "--metrics=", "--bogus"],
    # run: metrics, then the --gt and --pred checks, then --gt required
    [*RUN, "--metrics", ","], [*RUN, "--metrics", ""], [*RUN, "--metrics", " , ,"],
    [*RUN, "--metrics", " HOTA , Foo,bar,foo,,"], [*RUN, "--metrics", "TRACK_MAP,Jf,zz,ZZ,É,é"],
    [*RUN, "--metrics", "hota\x1f," + chr(0xA0) + "clear" + chr(0x3000) + ",x" + chr(0x200B)],
    [*RUN, "--metrics", "unknown", "--gt", "missing"],
    ["run", "--pred", "missing"], ["run", "--pred", "file.txt"], ["run", "--pred", "pred/"],
    ["run", "--pred", "./pred//."], ["run", "--pred", ""], ["run", "--pred="],
    ["run", "--pred", "-1"], ["run", "--pred", "-1.5"], ["run", "--pred", "-a b"],
    ["run", "--pred", "prédiction/日本"], ["run", "--pred", "pred", "--pred", "missing"],
    ["run", "--pre=pred", "--metr", "x"], ["run", "--pred", "pred", "--format", "mots"],
    [*RUN, "--gt", "missing-gt"], [*RUN, "--gt", "file.txt"], [*RUN, "--gt", "missing/"],
    ["run", "--gt", "gt", "--pred", "missing"], [*RUN, "--split", "test"],
    ["--", "run", "--pred", "pred"], [*RUN, "--metrics", "hota,hota"],
    ["run", "--dataset", "x", "--gt", "missing", "--pred", "pred"],
    ["run", "--dataset", "dancetrack", "--pred", "missing", "--gt", "missing"],
    # tests/test_cli.py cases that need no dataset and no metric
    ["run", "--dataset", "toy", "--pred", "pred", "--metrics", "hota,not-a-metric"],
    ["run", "--dataset", "toy", "--pred", "missing-predictions"],
    ["run", "--dataset", "dancetrack", "--pred", ".", "--gt", "missing-ground-truth"],
    ["run", "--pred", "."],
]  # fmt: skip


class ReachedLoad(Exception):
    pass


def _refuse_load(*args, **kwargs):
    raise ReachedLoad


def run_case(argv: list[str]) -> dict:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code
        except ReachedLoad:
            raise AssertionError(f"{argv}: reaches dataset loading") from None
    assert isinstance(code, int), (argv, code)
    return {"argv": argv, "exit": code, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


def main() -> None:
    os.environ["COLUMNS"] = "80"
    for name in ("NO_COLOR", "FORCE_COLOR", "PYTHON_COLORS"):
        os.environ.pop(name, None)
    # `_load` itself runs: it raises "--gt is required" before it calls a loader.
    for loader in ("load_dataset", "load_motchallenge", "load_mots"):
        setattr(cli, loader, _refuse_load)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        for name in FIXTURE["dirs"]:
            Path(tmp, name).mkdir()
        for name in FIXTURE["files"]:
            Path(tmp, name).write_text("")
        os.chdir(tmp)
        try:
            cases = [run_case(argv) for argv in CASES]
        finally:
            os.chdir(cwd)
    data = {"format": 1, "python": sys.version.split()[0], "fixture": FIXTURE, "cases": cases}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
