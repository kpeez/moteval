"""AnimalTrack: MOTChallenge-style GT rows, but no seqinfo.ini and split
membership comes from an external list file rather than a per-split directory.

Layout: ``<root>/gt_all/<seq>_gt.txt`` (GT for every sequence, flat) plus
``<root>/train_test_splits/videos_<split>.txt`` (one ``<seq>.mp4`` name per line)
for the "train"/"test" splits; "all" enumerates every ``gt_all/*_gt.txt`` file.
moteval never reads the accompanying ``<root>/frames_all/<seq>/*.jpg`` frames --
evaluation doesn't need them. GT frames are already 1-indexed, matching
`Track.frame`'s convention, so no shift is needed. Sequence length has no
seqinfo.ini source; it is derived from the last annotated frame (see the same
note in `bft.py`).
"""

from pathlib import Path

from moteval.benchmarks.motchallenge import (
    MOTChallengeConfig,
    load_layout,
    max_frame_seq_length,
)
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats import Track

ANIMALTRACK_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
ANIMALTRACK_PROTOCOL = Protocol(
    name="animaltrack",
    frame_convention=ANIMALTRACK_CONVENTION,
    eval_classes=(1,),
)


def _animaltrack_seq_names(base: Path, split: str) -> list[str]:
    if split == "all":
        gt_dir = base / "gt_all"
        if not gt_dir.is_dir():
            raise ValueError(f"split directory not found: {gt_dir}")
        return sorted(p.name.removesuffix("_gt.txt") for p in gt_dir.glob("*_gt.txt"))
    if split not in ("train", "test"):
        raise ValueError(f"unknown animaltrack split {split!r}; expected 'all', 'train', or 'test'")
    split_file = base / "train_test_splits" / f"videos_{split}.txt"
    if not split_file.is_file():
        raise ValueError(f"animaltrack split {split!r} not found at {split_file}")
    return [
        line.strip().removesuffix(".mp4")
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]


def _animaltrack_gt_path(base: Path, split: str, seq_name: str) -> Path:
    return base / "gt_all" / f"{seq_name}_gt.txt"


def _animaltrack_seq_length(
    base: Path, split: str, seq_name: str, tracks: tuple[Track, ...]
) -> int:
    return max_frame_seq_length(seq_name, tracks)


ANIMALTRACK_CONFIG = MOTChallengeConfig(
    name="animaltrack",
    default_root=Path("data/benchmarks/animaltrack"),
    protocol=ANIMALTRACK_PROTOCOL,
    seq_names=_animaltrack_seq_names,
    gt_path=_animaltrack_gt_path,
    seq_length=_animaltrack_seq_length,
)


def load_animaltrack(root: str | Path | None = None, split: str = "all") -> MOTDataset[GtSequence]:
    return load_layout(ANIMALTRACK_CONFIG, root=root, split=split)
