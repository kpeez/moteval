"""Shared test fixtures: the in-memory toy dataset and a two-class dataset.

Two tiny 1-indexed MOTChallenge-style sequences, two tracks each over five
frames. Ground truth is generated in memory so tests stay hermetic;
predictions are read from a ``<seq>.txt`` directory by ``evaluate``;
`write_perfect_predictions` writes GT boxes under prediction-owned track ids
(AGENTS.md: predictions are numbered independently of GT). Tests that
resolve the dataset by name (CLI ``--dataset toy``) request the ``toy_benchmark``
fixture, which registers a loader in `BENCHMARKS` for that test only. That loader
takes the required ``root`` and the ``split`` keyword like every benchmark loader, and
ignores both — the data is synthesized, not read from disk.

`load_two_class` is a one-sequence dataset whose protocol evaluates classes 1 and 2;
`write_two_class_predictions` writes its predictions with the class in column 8, and
the ``two_class_benchmark`` fixture registers it as ``two-class``.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from moteval.benchmarks import BENCHMARKS
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats import Track, write_mot

TOY_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
TOY_PROTOCOL = Protocol(
    name="toy",
    frame_convention=TOY_CONVENTION,
    eval_classes=(1,),
)


def _linear_track(
    track_id: int, x0: float, y0: float, dx: float, w: float, h: float
) -> list[Track]:
    return [
        Track(frame=f, track_id=track_id, x=x0 + dx * (f - 1), y=y0, w=w, h=h, conf=1.0)
        for f in range(1, 6)
    ]


def load_toy() -> MOTDataset[GtSequence]:
    seq1 = GtSequence(
        name="toy-0001",
        num_timesteps=5,
        tracks=tuple(_linear_track(1, 10, 10, 2, 20, 20) + _linear_track(2, 100, 100, 2, 30, 40)),
    )
    seq2 = GtSequence(
        name="toy-0002",
        num_timesteps=5,
        tracks=tuple(_linear_track(1, 50, 50, 5, 25, 25) + _linear_track(2, 200, 30, 0, 40, 40)),
    )
    return MOTDataset(
        name="toy",
        split="val",
        sequences=(seq1, seq2),
        protocol=TOY_PROTOCOL,
    )


def _toy_loader(root: str | Path, split: str = "val") -> MOTDataset[GtSequence]:
    return load_toy()


# Prediction ids differ from the GT ids (1, 2) and reverse their order.
_PERFECT_PRED_IDS = {1: 9, 2: 7}


def write_perfect_predictions(dataset: MOTDataset[GtSequence], pred_dir: Path) -> None:
    """Write each toy sequence's GT boxes as predictions under their own track ids.

    Only the toy GT ids (1, 2) are mapped; any other id raises `KeyError`.
    """
    for seq in dataset.sequences:
        preds = [replace(t, track_id=_PERFECT_PRED_IDS[t.track_id]) for t in seq.tracks]
        write_mot(pred_dir / f"{seq.name}.txt", preds)


@pytest.fixture
def toy_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(BENCHMARKS, "toy", _toy_loader)


TWO_CLASS_SEQUENCE = "two-class-0001"


def load_two_class() -> MOTDataset[GtSequence]:
    """One 2-frame sequence of 10x10 boxes that never overlap each other.

    Class 1: GT track 10 at (0, 0). Class 2: GT tracks 20 at (100, 100) and 30 at
    (200, 200). Every track spans both frames.
    """
    tracks = tuple(
        Track(frame=f, track_id=tid, x=x, y=x, w=10, h=10, conf=1.0, class_id=cls)
        for tid, x, cls in [(10, 0.0, 1), (20, 100.0, 2), (30, 200.0, 2)]
        for f in (1, 2)
    )
    return MOTDataset(
        name="two-class",
        split="val",
        sequences=(GtSequence(name=TWO_CLASS_SEQUENCE, num_timesteps=2, tracks=tracks),),
        protocol=replace(TOY_PROTOCOL, name="two-class", eval_classes=(1, 2)),
    )


def write_two_class_predictions(pred_dir: Path) -> None:
    """Predictions for `load_two_class`, class in column 8, every IoU exactly 1 or 0.

    Class 1: track 100 covers GT 10 in both frames; track 300 sits on GT 30 (a class-2
    box) in frame 1, so it is a class-1 false positive. Class 2: track 200 covers
    GT 20 in both frames; GT 30 is missed.
    """
    rows = [
        "1,100,0,0,10,10,1,1",
        "2,100,0,0,10,10,1,1",
        "1,300,200,200,10,10,1,1",
        "1,200,100,100,10,10,1,2",
        "2,200,100,100,10,10,1,2",
    ]
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / f"{TWO_CLASS_SEQUENCE}.txt").write_text("\n".join(rows) + "\n")


def _two_class_loader(root: str | Path, split: str = "val") -> MOTDataset[GtSequence]:
    return load_two_class()


@pytest.fixture
def two_class_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(BENCHMARKS, "two-class", _two_class_loader)
