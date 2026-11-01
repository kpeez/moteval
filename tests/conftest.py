"""Shared test helpers: the in-memory toy dataset and a two-class dataset.

Two tiny 1-indexed MOTChallenge-style sequences, two tracks each over five
frames. Ground truth is generated in memory so tests stay hermetic;
predictions are read from a ``<seq>.txt`` directory by ``evaluate``;
`write_perfect_predictions` writes GT boxes under prediction-owned track ids
(AGENTS.md: predictions are numbered independently of GT). The CLI runs in Rust and
cannot see a dataset built in Python, so CLI tests write the toy dataset to disk with
`write_toy_layout` and read it with ``--gt``.

`load_two_class` is a one-sequence dataset whose protocol evaluates classes 1 and 2;
`write_two_class_predictions` writes its predictions with the class in column 8.
"""

from dataclasses import replace
from pathlib import Path

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


def write_toy_layout(root: Path) -> None:
    """Write `load_toy` in the MOTChallenge layout under ``root`` (split ``val``):
    ``load_motchallenge(root, "val")`` reads the same sequences and tracks."""
    for seq in load_toy().sequences:
        seq_dir = root / "val" / seq.name
        write_mot(seq_dir / "gt" / "gt.txt", list(seq.tracks))
        (seq_dir / "seqinfo.ini").write_text(f"[Sequence]\nseqLength={seq.num_timesteps}\n")


# Prediction ids differ from the GT ids (1, 2) and reverse their order.
_PERFECT_PRED_IDS = {1: 9, 2: 7}


def write_perfect_predictions(dataset: MOTDataset[GtSequence], pred_dir: Path) -> None:
    """Write each toy sequence's GT boxes as predictions under their own track ids.

    Only the toy GT ids (1, 2) are mapped; any other id raises `KeyError`.
    """
    for seq in dataset.sequences:
        preds = [replace(t, track_id=_PERFECT_PRED_IDS[t.track_id]) for t in seq.tracks]
        write_mot(pred_dir / f"{seq.name}.txt", preds)


def write_flawed_predictions(dataset: MOTDataset[GtSequence], pred_dir: Path) -> None:
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
