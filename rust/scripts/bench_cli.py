"""Time the three moteval CLIs on real benchmarks and check that they agree.

The three entry points:

- ``reference``: the pure-Python CLI of the ``py-reference`` tag, from the reference
  venv (``artifacts/rust-port/ref-venv/bin/moteval``);
- ``python``: the Rust-backed Python package's console script (``.venv/bin/moteval``);
- ``rust``: the Rust binary (``rust/target/release/moteval``; build it first with
  ``cargo build --release``).

Each case scores the seeded predictions ``dump_loaders.py`` wrote
(``<testdata>/loaders_real/<case>/pred``) with every metric that applies. Before timing,
each CLI runs once with ``--out-json``, and the three JSON files must be byte-identical.
Wall time comes from ``hyperfine --warmup 3 --runs 10 --export-json``; peak RSS is the
largest "Maximum resident set size" of three ``/usr/bin/time -v`` runs. Raw output goes
to ``--out`` (default ``artifacts/rust-port/bench``), with a ``summary.json``.

    MOTEVAL_DATA_ROOT=/data/tracking/external \\
        python rust/scripts/bench_cli.py --testdata artifacts/rust-port/testdata

Standard library only; ``hyperfine`` comes from ``mise.toml``.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOX_METRICS = "hota,clear,identity,count,track_map"
MASK_METRICS = "hota,clear,identity,count,jf"
# (case, dataset, split, metrics)
CASES = (
    ("dancetrack_val", "dancetrack", "val", BOX_METRICS),
    ("sportsmot_val", "sportsmot", "val", BOX_METRICS),
    ("mots20_train", "mots20", "train", MASK_METRICS),
    ("chimpact_val", "chimpact", "val", BOX_METRICS),
)
CLIS = {
    "reference": REPO_ROOT / "artifacts" / "rust-port" / "ref-venv" / "bin" / "moteval",
    "python": REPO_ROOT / ".venv" / "bin" / "moteval",
    "rust": REPO_ROOT / "rust" / "target" / "release" / "moteval",
}
_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")


def run_args(dataset: str, split: str, metrics: str, pred: Path) -> list[str]:
    return [
        "run",
        "--dataset",
        dataset,
        "--split",
        split,
        "--pred",
        str(pred),
        "--metrics",
        metrics,
    ]


def peak_rss_kib(command: list[str], runs: int, log: Path) -> int:
    """The largest maximum RSS of ``runs`` runs, in KiB; every run's report goes to log."""
    peaks = []
    with log.open("w") as out:
        for _ in range(runs):
            done = subprocess.run(
                ["/usr/bin/time", "-v", *command], capture_output=True, text=True, check=True
            )
            out.write(done.stderr)
            match = _RSS.search(done.stderr)
            if match is None:
                raise SystemExit(f"no RSS line from /usr/bin/time for {command}")
            peaks.append(int(match.group(1)))
    return max(peaks)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--testdata", type=Path, required=True, help="MOTEVAL_RUST_TESTDATA")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/rust-port/bench")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--rss-runs", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("MOTEVAL_DATA_ROOT"):
        raise SystemExit("set MOTEVAL_DATA_ROOT")
    for name, path in CLIS.items():
        if not path.is_file():
            raise SystemExit(f"{name} CLI not found: {path}")
    hyperfine = shutil.which("hyperfine")
    if hyperfine is None:
        raise SystemExit("hyperfine not on PATH (mise exec -- python ...)")
    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}
    for case, dataset, split, metrics in CASES:
        pred = args.testdata / "loaders_real" / case / "pred"
        if not pred.is_dir():
            raise SystemExit(f"{pred} missing: run rust/scripts/dump_loaders.py")
        case_dir = args.out / case
        case_dir.mkdir(exist_ok=True)
        outputs = {}
        for name, cli in CLIS.items():
            out_json = case_dir / f"{name}.json"
            command = [
                str(cli),
                *run_args(dataset, split, metrics, pred),
                "--out-json",
                str(out_json),
            ]
            subprocess.run(command, check=True, capture_output=True)
            outputs[name] = out_json.read_bytes()
        identical = len(set(outputs.values())) == 1
        print(f"{case}: JSON outputs identical across CLIs: {identical}", flush=True)
        if not identical:
            raise SystemExit(f"{case}: the CLIs disagree; see {case_dir}")

        commands = {
            name: " ".join([str(cli), *run_args(dataset, split, metrics, pred)])
            for name, cli in CLIS.items()
        }
        hyperfine_json = case_dir / "hyperfine.json"
        subprocess.run(
            [
                hyperfine,
                "--warmup",
                str(args.warmup),
                "--runs",
                str(args.runs),
                "--export-json",
                str(hyperfine_json),
                "--output",
                "null",
                *[arg for name, command in commands.items() for arg in ("-n", name, command)],
            ],
            check=True,
        )
        timing = {r["command"]: r for r in json.loads(hyperfine_json.read_text())["results"]}
        rss = {
            name: peak_rss_kib(
                [str(cli), *run_args(dataset, split, metrics, pred)],
                args.rss_runs,
                case_dir / f"time-{name}.log",
            )
            for name, cli in CLIS.items()
        }
        summary[case] = {
            "metrics": metrics,
            "json_identical": identical,
            "mean_s": {name: timing[name]["mean"] for name in CLIS},
            "stddev_s": {name: timing[name]["stddev"] for name in CLIS},
            "peak_rss_kib": rss,
        }
        print(json.dumps({case: summary[case]}, indent=2), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out / 'summary.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
