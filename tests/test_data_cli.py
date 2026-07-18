import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXPECTED_BENCHMARKS = (
    "animaltrack",
    "bft",
    "chimpact",
    "dancetrack",
    "gmot40",
    "mots20",
    "panaf500",
    "sportsmot",
    "toy",
    "uavdt",
)


def _run_data(*args: object, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    entry_point = ROOT / ".venv" / "bin" / "moteval"
    return subprocess.run(
        [str(entry_point), "data", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | (env or {}),
    )


def test_data_list_reports_every_supported_benchmark() -> None:
    completed = _run_data("list")

    assert completed.returncode == 0, completed.stderr
    rows = [line.split() for line in completed.stdout.splitlines()[1:]]
    assert tuple(row[0] for row in rows) == EXPECTED_BENCHMARKS
    assert dict(rows)["chimpact"] == "unfetchable"
    assert dict(rows)["toy"] == "bundled"


def test_data_status_uses_loader_layout_markers_and_bundled_toy(tmp_path: Path) -> None:
    (tmp_path / "animaltrack" / "gt_all").mkdir(parents=True)
    (tmp_path / "chimpact" / "ChimpACT_release_v1" / "labels").mkdir(parents=True)

    completed = _run_data("status", "--root", tmp_path)

    assert completed.returncode == 0, completed.stderr
    rows = {
        parts[0]: parts[1] for line in completed.stdout.splitlines()[1:] if (parts := line.split())
    }
    assert rows["animaltrack"] == "present"
    assert rows["chimpact"] == "present"
    assert rows["dancetrack"] == "absent"
    assert rows["toy"] == "bundled"


def test_data_root_option_overrides_environment(tmp_path: Path) -> None:
    environment_root = tmp_path / "environment"
    cli_root = tmp_path / "cli"
    (environment_root / "gmot40" / "track_label").mkdir(parents=True)

    completed = _run_data(
        "--root",
        cli_root,
        "status",
        env={"MOTEVAL_DATA_ROOT": str(environment_root)},
    )

    assert completed.returncode == 0, completed.stderr
    gmot40_row = next(line for line in completed.stdout.splitlines() if line.startswith("gmot40"))
    assert "absent" in gmot40_row
    assert str(cli_root / "gmot40") in gmot40_row


def test_data_status_uses_environment_root(tmp_path: Path) -> None:
    (tmp_path / "sportsmot" / "val").mkdir(parents=True)

    completed = _run_data("status", env={"MOTEVAL_DATA_ROOT": str(tmp_path)})

    assert completed.returncode == 0, completed.stderr
    sportsmot_row = next(
        line for line in completed.stdout.splitlines() if line.startswith("sportsmot")
    )
    assert "present" in sportsmot_row
    assert str(tmp_path / "sportsmot") in sportsmot_row


def test_unknown_download_lists_valid_benchmark_names(tmp_path: Path) -> None:
    completed = _run_data("download", "not-a-benchmark", "--root", tmp_path)

    assert completed.returncode != 0
    assert "unknown benchmark 'not-a-benchmark'" in completed.stderr
    assert "valid names: " + ", ".join(EXPECTED_BENCHMARKS) in completed.stderr
    assert "Traceback" not in completed.stderr


def test_unfetchable_download_gives_manual_acquisition_layout(tmp_path: Path) -> None:
    completed = _run_data("download", "chimpact", "--root", tmp_path)

    assert completed.returncode != 0
    assert "no stable programmatic artifact" in completed.stderr
    assert "https://github.com/ShirleyMaxx/ChimpACT" in completed.stderr
    assert "<root>/chimpact/ChimpACT_release_v1/labels/<clip>.json" in completed.stderr
    assert "Traceback" not in completed.stderr
