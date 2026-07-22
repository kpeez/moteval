"""Tests for the dev download script (scripts/download_benchmarks.py).

Not a package module, so it is loaded here by file path. Fetchers are
monkeypatched; only the ``network``-marked test touches the real internet.
"""

import importlib.util
import io
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "download_benchmarks.py"
_spec = importlib.util.spec_from_file_location("download_benchmarks", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download)


def _write_zip(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _write_tar(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            payload = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_dancetrack_download_remaps_sharded_split_archives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fetch_snapshot(_step: download.HFSnapshot, destination: Path) -> None:
        for archive_name, prefix, sequence in (
            ("train1.zip", "train1", "dancetrack0001"),
            ("train2.zip", "train2", "dancetrack0002"),
            ("val.zip", "val", "dancetrack0010"),
            ("test1.zip", "test1", "dancetrack0020"),
            ("test2.zip", "test2", "dancetrack0021"),
        ):
            _write_zip(
                destination / archive_name,
                {
                    f"{prefix}/{sequence}/seqinfo.ini": "[Sequence]\nseqLength=1\n",
                    f"{prefix}/{sequence}/gt/gt.txt": "1,1,0,0,1,1,1,-1,-1,-1\n",
                },
            )

    monkeypatch.setattr(download, "_fetch_hf_snapshot", fetch_snapshot)

    destination = download.download_benchmark("dancetrack", tmp_path)

    assert (destination / "train" / "dancetrack0001" / "gt" / "gt.txt").is_file()
    assert (destination / "train" / "dancetrack0002" / "seqinfo.ini").is_file()
    assert (destination / "val" / "dancetrack0010" / "gt" / "gt.txt").is_file()
    assert (destination / "test" / "dancetrack0021" / "seqinfo.ini").is_file()
    assert not (destination / "train1").exists()


def test_sportsmot_download_flattens_pinned_tar_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fetch_snapshot(_step: download.HFSnapshot, destination: Path) -> None:
        for split in ("train", "val", "test"):
            prefix = f"sportsmot_publish/dataset/{split}/{split}-seq"
            _write_tar(
                destination / "dataset" / f"{split}.tar",
                {
                    f"{prefix}/seqinfo.ini": "[Sequence]\nseqLength=1\n",
                    f"{prefix}/gt/gt.txt": "1,1,0,0,1,1,1,-1,-1,-1\n",
                },
            )

    monkeypatch.setattr(download, "_fetch_hf_snapshot", fetch_snapshot)

    destination = download.download_benchmark("sportsmot", tmp_path)

    for split in ("train", "val", "test"):
        assert (destination / split / f"{split}-seq" / "seqinfo.ini").is_file()
        assert (destination / split / f"{split}-seq" / "gt" / "gt.txt").is_file()
    assert not (destination / "sportsmot_publish").exists()


def test_mots20_download_flattens_enclosing_archive_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fetch_archive(_step: download.HttpFile, destination: Path) -> None:
        _write_zip(
            destination / "MOTS.zip",
            {
                "MOTS/train/MOTS20-02/seqinfo.ini": "[Sequence]\nseqLength=1\n",
                "MOTS/train/MOTS20-02/gt/gt.txt": "1 1 2 1 1 0\n",
                "MOTS/test/MOTS20-01/seqinfo.ini": "[Sequence]\nseqLength=1\n",
            },
        )

    monkeypatch.setattr(download, "_fetch_http", fetch_archive)

    destination = download.download_benchmark("mots20", tmp_path)

    assert (destination / "train" / "MOTS20-02" / "gt" / "gt.txt").is_file()
    assert (destination / "train" / "MOTS20-02" / "seqinfo.ini").is_file()
    assert (destination / "test" / "MOTS20-01" / "seqinfo.ini").is_file()
    assert not (destination / "MOTS" / "train").exists()


def test_gmot40_download_extracts_track_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fetch_file(step: download.HttpFile, destination: Path) -> None:
        if step.filename == "track_label.zip":
            _write_zip(
                destination / step.filename,
                {"track_label/airplane-0.txt": "0,1,0,0,1,1,1,-1,-1,-1\n"},
            )
        else:
            (destination / step.filename).write_bytes(b"images archive fixture")

    monkeypatch.setattr(download, "_fetch_http", fetch_file)

    destination = download.download_benchmark("gmot40", tmp_path)

    assert (destination / "track_label" / "airplane-0.txt").is_file()


def test_uavdt_download_extracts_gt_and_ignore_regions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fetch_file(step: download.DriveFile, destination: Path) -> None:
        if step.filename == "UAV-benchmark-MOTD_v1.0.zip":
            _write_zip(
                destination / step.filename,
                {
                    "UAV-benchmark-MOTD_v1.0/GT/M0101_gt.txt": "1,1,0,0,1,1,1,1,1\n",
                    "UAV-benchmark-MOTD_v1.0/GT/M0101_gt_ignore.txt": ("1,-1,0,0,1,1,1,1,1\n"),
                },
            )
        else:
            (destination / step.filename).write_bytes(b"images archive fixture")

    monkeypatch.setattr(download, "_fetch_drive_chunked", fetch_file)

    destination = download.download_benchmark("uavdt", tmp_path)

    gt_root = destination / "UAV-benchmark-MOTD_v1.0" / "GT"
    assert (gt_root / "M0101_gt.txt").is_file()
    assert (gt_root / "M0101_gt_ignore.txt").is_file()


@pytest.mark.parametrize(
    ("benchmark", "nested_marker", "expected_marker"),
    [
        ("animaltrack", "AnimalTrack/gt_all", "gt_all"),
        ("bft", "BFT/annotations_mot", "annotations_mot"),
    ],
)
def test_gdrive_downloads_are_normalized_to_benchmark_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    benchmark: str,
    nested_marker: str,
    expected_marker: str,
) -> None:
    def fetch_folder(_step: download.GDriveFolder, destination: Path) -> None:
        marker = destination / nested_marker
        marker.mkdir(parents=True)
        (marker / "fixture.txt").write_text("gt")

    monkeypatch.setattr(download, "_fetch_gdrive_folder", fetch_folder)

    destination = download.download_benchmark(benchmark, tmp_path)

    assert (destination / expected_marker / "fixture.txt").is_file()


def test_unfetchable_benchmark_fails_before_creating_a_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="official repository only links a gated Google Form"):
        download.download_benchmark("chimpact", tmp_path)

    assert not (tmp_path / "chimpact").exists()


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


def test_script_list_reports_every_supported_benchmark(capsys):
    exit_code = download.main(["list"])

    assert exit_code == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    assert tuple(row[0] for row in rows) == _EXPECTED_BENCHMARKS
    assert dict(rows)["chimpact"] == "unfetchable"


def test_script_status_uses_loader_layout_markers(tmp_path, capsys):
    (tmp_path / "animaltrack" / "gt_all").mkdir(parents=True)
    (tmp_path / "chimpact" / "ChimpACT_release_v1" / "labels").mkdir(parents=True)

    exit_code = download.main(["--root", str(tmp_path), "status"])

    assert exit_code == 0
    rows = {
        parts[0]: parts[1]
        for line in capsys.readouterr().out.splitlines()[1:]
        if (parts := line.split())
    }
    assert rows["animaltrack"] == "present"
    assert rows["chimpact"] == "present"
    assert rows["dancetrack"] == "absent"


def test_script_root_option_overrides_environment(tmp_path, monkeypatch, capsys):
    environment_root = tmp_path / "environment"
    cli_root = tmp_path / "cli"
    (environment_root / "gmot40" / "track_label").mkdir(parents=True)
    monkeypatch.setenv("MOTEVAL_DATA_ROOT", str(environment_root))

    exit_code = download.main(["--root", str(cli_root), "status"])

    assert exit_code == 0
    gmot40_row = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("gmot40")
    )
    assert "absent" in gmot40_row
    assert str(cli_root / "gmot40") in gmot40_row


def test_script_status_uses_environment_root(tmp_path, monkeypatch, capsys):
    (tmp_path / "sportsmot" / "val").mkdir(parents=True)
    monkeypatch.setenv("MOTEVAL_DATA_ROOT", str(tmp_path))

    exit_code = download.main(["status"])

    assert exit_code == 0
    sportsmot_row = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("sportsmot")
    )
    assert "present" in sportsmot_row
    assert str(tmp_path / "sportsmot") in sportsmot_row


def test_script_unknown_download_lists_valid_benchmark_names(tmp_path, capsys):
    with pytest.raises(SystemExit):
        download.main(["--root", str(tmp_path), "download", "not-a-benchmark"])

    err = capsys.readouterr().err
    assert "unknown benchmark 'not-a-benchmark'" in err
    assert "valid names: " + ", ".join(_EXPECTED_BENCHMARKS) in err
    assert "Traceback" not in err


def test_script_unfetchable_download_gives_manual_acquisition_layout(tmp_path, capsys):
    with pytest.raises(SystemExit):
        download.main(["--root", str(tmp_path), "download", "chimpact"])

    err = capsys.readouterr().err
    assert "no stable programmatic artifact" in err
    assert "https://github.com/ShirleyMaxx/ChimpACT" in err
    assert "<root>/chimpact/ChimpACT_release_v1/labels/<clip>.json" in err


@pytest.mark.network
def test_gmot40_track_labels_download_and_extract_from_github(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    selected_marks = request.config.getoption("-m")
    if os.environ.get("RUN_NETWORK_TESTS") != "1" and "network" not in selected_marks:
        pytest.skip("set RUN_NETWORK_TESTS=1 or select -m network")

    destination = tmp_path / "gmot40"
    destination.mkdir()
    label_step = next(
        step
        for step in download.SPECS["gmot40"].fetch_steps
        if isinstance(step, download.HttpFile) and step.filename == "track_label.zip"
    )
    for step in (label_step, download.ZipExtraction("track_label.zip")):
        download._execute_step(step, destination)

    labels = sorted((destination / "track_label").glob("*.txt"))
    assert len(labels) == 40
    assert all(label.stat().st_size > 0 for label in labels)
