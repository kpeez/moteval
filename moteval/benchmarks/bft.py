"""BFT: MOTChallenge-style GT rows, but no seqinfo.ini and GT lives in a flat
per-split directory rather than nested under each sequence.

Layout: ``<root>/annotations_mot/<split>/<seq>.txt`` (GT only; moteval never reads
the accompanying ``<root>/<split>/<seq>/*.jpg`` frames -- evaluation doesn't need
them). Frames are already 1-indexed on disk, matching `Track.frame`'s convention,
so no shift is needed. Sequence length has no seqinfo.ini source; it is derived
from the last annotated frame, so a sequence with no GT in its final frames would
undercount -- harmless for metrics (empty trailing frames contribute nothing) but
would reject predictions that fall past the last annotated frame.
"""

from pathlib import Path

from moteval.benchmarks.base import register_dataset
from moteval.benchmarks.motchallenge import (
    MOTChallengeConfig,
    load_motchallenge,
    max_frame_seq_length,
)
from moteval.data.model import FrameConvention, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import Track

BFT_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
BFT_PROTOCOL = Protocol(
    name="bft",
    frame_convention=BFT_CONVENTION,
    eval_classes=(1,),
)


def _bft_seq_names(base: Path, split: str) -> list[str]:
    ann_dir = base / "annotations_mot" / split
    if not ann_dir.is_dir():
        raise ValueError(f"split directory not found: {ann_dir}")
    return sorted(p.stem for p in ann_dir.glob("*.txt"))


def _bft_gt_path(base: Path, split: str, seq_name: str) -> Path:
    return base / "annotations_mot" / split / f"{seq_name}.txt"


def _bft_seq_length(base: Path, split: str, seq_name: str, tracks: tuple[Track, ...]) -> int:
    return max_frame_seq_length(seq_name, tracks)


BFT_CONFIG = MOTChallengeConfig(
    name="bft",
    default_root=Path("data/benchmarks/bft"),
    protocol=BFT_PROTOCOL,
    seq_names=_bft_seq_names,
    gt_path=_bft_gt_path,
    seq_length=_bft_seq_length,
)


def load_bft(root: str | Path | None = None, split: str = "val") -> MOTDataset:
    return load_motchallenge(BFT_CONFIG, root=root, split=split)


register_dataset("bft")(load_bft)
