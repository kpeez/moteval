"""Real-data gate for the benchmarks that have no TrackEval parity oracle.

BFT, AnimalTrack, GMOT-40, UAVDT, PanAf500 and ChimpACT ship no TrackEval loader, so
their scores cannot be checked against frozen upstream numbers. Each test here instead
submits the benchmark's own ground truth as predictions. The predictions come from the
raw annotation files, never from the loader. CLEAR must count every submitted row as an
exact match (MOTP 1.0), with no false positives or identity switches, and each split
must load its published number of sequences.

PanAf500 and ChimpACT convert JSON, so their tests check the loader's frame, id and box
mapping. BFT, AnimalTrack, GMOT-40 and UAVDT already store MOTChallenge rows, which both
sides parse with `read_mot`; for them the tests check the split's sequence list, the
declared frame convention and the derived sequence length. The DanceTrack and SportsMOT
parity tests own `read_mot` itself.

ChimpACT submits its labelled keyframes only, because its ground truth fills the frames
between them. Its loader documents that every keyframe box fills exactly the 9 frames
after it (interpolated toward the next keyframe, or held when the track has none), so
the misses must be exactly 9 per submitted row. That rule is the loader's deliberate
match to the legacy track-zoo loader, not something the raw labels prove; the synthetic
tests in `test_loaders.py` pin the interpolated box values. Every other benchmark has
no misses.

Each test skips loudly when its dataset is absent from the data root
(`moteval.benchmarks.default_data_root`).
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from moteval import CLEAR, evaluate, load_dataset
from moteval.benchmarks import default_data_root


def _raw_rows(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _bft_rows(root: Path, split: str, seq: str) -> list[str]:
    return _raw_rows(root / "annotations_mot" / split / f"{seq}.txt")


def _animaltrack_rows(root: Path, split: str, seq: str) -> list[str]:
    return _raw_rows(root / "gt_all" / f"{seq}_gt.txt")


def _gmot40_rows(root: Path, split: str, seq: str) -> list[str]:
    return _raw_rows(root / "track_label" / f"{seq}.txt")


def _uavdt_rows(root: Path, split: str, seq: str) -> list[str]:
    # Column 7 is UAVDT's score flag. The MOTD README says a 0 box is ignored, and the
    # official evaluateTracking.m deletes those GT rows before matching.
    rows = _raw_rows(root / "UAV-benchmark-MOTD_v1.0" / "GT" / f"{seq}_gt.txt")
    return [row for row in rows if float(row.split(",")[6]) != 0]


def _panaf500_rows(root: Path, split: str, seq: str) -> list[str]:
    data = json.loads((root / "annotations" / split / f"{seq}.json").read_text())
    rows = []
    for frame in data["annotations"]:
        for det in frame["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            rows.append(f"{frame['frame_id']},{det['ape_id']},{x1},{y1},{x2 - x1},{y2 - y1},1")
    return rows


def _chimpact_keyframe_rows(root: Path, split: str, seq: str) -> list[str]:
    # An image's file_name stem is a keyframe block; block N is video frame N * 10.
    # bbox_id 23 is the official converter's unnamed catch-all track, which it drops.
    labels = json.loads((root / "ChimpACT_release_v1" / "labels" / f"{seq}.json").read_text())
    frame_of = {img["id"]: int(Path(img["file_name"]).stem) * 10 for img in labels["images"]}
    return [
        f"{frame_of[ann['image_id']]},{ann['bbox_id']},{','.join(map(str, ann['bbox']))},1"
        for ann in labels["annotations"]
        if ann["bbox_id"] != 23
    ]


RawRows = Callable[[Path, str, str], list[str]]

# (benchmark, split, raw-row writer, published sequence count, misses per row). The
# counts are each release's split sizes: GMOT-40 is 10 categories of 4 sequences, its
# animal subset 4 of those categories; ChimpACT's come from the official split lists.
CASES: list[tuple[str, str, RawRows, int, int]] = [
    ("bft", "train", _bft_rows, 45, 0),
    ("bft", "val", _bft_rows, 25, 0),
    ("bft", "test", _bft_rows, 36, 0),
    ("animaltrack", "all", _animaltrack_rows, 58, 0),
    ("animaltrack", "train", _animaltrack_rows, 32, 0),
    ("animaltrack", "test", _animaltrack_rows, 26, 0),
    ("gmot40", "test", _gmot40_rows, 40, 0),
    ("gmot40", "animal", _gmot40_rows, 16, 0),
    ("uavdt", "all", _uavdt_rows, 50, 0),
    ("panaf500", "train", _panaf500_rows, 400, 0),
    ("panaf500", "validation", _panaf500_rows, 25, 0),
    ("panaf500", "test", _panaf500_rows, 75, 0),
    ("chimpact", "train", _chimpact_keyframe_rows, 127, 9),
    ("chimpact", "val", _chimpact_keyframe_rows, 17, 9),
    ("chimpact", "test", _chimpact_keyframe_rows, 19, 9),
]


@pytest.mark.real_data
@pytest.mark.parametrize(
    ("name", "split", "raw_rows", "num_sequences", "misses_per_row"),
    CASES,
    ids=[f"{c[0]}-{c[1]}" for c in CASES],
)
def test_raw_ground_truth_scores_perfectly(
    tmp_path: Path,
    name: str,
    split: str,
    raw_rows: RawRows,
    num_sequences: int,
    misses_per_row: int,
) -> None:
    root = default_data_root() / name
    if not root.is_dir():
        fetch = (
            "place the release by hand (see docs/DATASETS.md)"
            if name == "chimpact"
            else f"fetch it with `scripts/download_benchmarks.py download {name}`"
        )
        pytest.skip(f"SKIPPING REAL-DATA LOADER GATE: {name} not found under {root} — {fetch}")
    dataset = load_dataset(name, root=root, split=split)
    assert len(dataset.sequences) == num_sequences
    submitted = 0
    for seq in dataset.sequences:
        rows = raw_rows(root, split, seq.name)
        (tmp_path / f"{seq.name}.txt").write_text("".join(f"{row}\n" for row in rows))
        submitted += len(rows)

    clear = evaluate(dataset, tmp_path, [CLEAR()]).combined["CLEAR"]

    counts = (clear["CLR_TP"], clear["CLR_FP"], clear["IDSW"], clear["CLR_FN"])
    assert counts == (submitted, 0, 0, misses_per_row * submitted)
    assert clear["MOTP"] == 1.0
