"""GMOT-40: MOTChallenge-style GT rows, natively 0-indexed, no seqinfo.ini, GT
in a flat directory keyed by sequence name.

Layout: ``<root>/track_label/<seq>.txt`` (GT only; moteval never reads the
accompanying ``<root>/GenericMOT_JPEG_Sequence/<seq>/img1/*.jpg`` frames --
evaluation doesn't need them). No train/val/test split exists on disk: "test"
exposes all 40 sequences (GMOT-40 is used test-only); "animal" exposes the
16-sequence GMOT-40-Animal subset (picked by category-name substring) that the
spec's HOTA 62.4 baseline refers to.

GT frame numbers on disk are 0-indexed. Unlike the legacy track-zoo loader
(which shifted GT frames +1 at load time to match a hardcoded 1-indexed
convention), this loader keeps the raw 0-indexed frame numbers and declares
`FrameConvention(first_frame=0)` -- frame-indexing is the declared loader
parameter (ADR-0002), not a load-time rewrite. Sequence length has no
seqinfo.ini source; it is derived from the last annotated frame (see the same
note in `bft.py`), plus one since frame 0 is the first timestep.
"""

from pathlib import Path

from moteval.benchmarks.base import register_dataset
from moteval.benchmarks.motchallenge import (
    MOTChallengeConfig,
    load_motchallenge,
    max_frame_seq_length,
)
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import Track

_ANIMAL_SUBSTRINGS = ("bird", "fish", "insect", "stock")

GMOT40_CONVENTION = FrameConvention(name="0-indexed", first_frame=0)
GMOT40_PROTOCOL = Protocol(
    name="gmot40",
    frame_convention=GMOT40_CONVENTION,
    eval_classes=(1,),
)


def _gmot40_seq_names(base: Path, split: str) -> list[str]:
    gt_dir = base / "track_label"
    if not gt_dir.is_dir():
        raise ValueError(f"split directory not found: {gt_dir}")
    names = sorted(p.stem for p in gt_dir.glob("*.txt"))
    if split == "animal":
        return [name for name in names if any(s in name for s in _ANIMAL_SUBSTRINGS)]
    if split != "test":
        raise ValueError(f"unknown gmot40 split {split!r}; expected 'test' or 'animal'")
    return names


def _gmot40_gt_path(base: Path, split: str, seq_name: str) -> Path:
    return base / "track_label" / f"{seq_name}.txt"


def _gmot40_seq_length(base: Path, split: str, seq_name: str, tracks: tuple[Track, ...]) -> int:
    return max_frame_seq_length(seq_name, tracks, offset=1)


GMOT40_CONFIG = MOTChallengeConfig(
    name="gmot40",
    default_root=Path("data/benchmarks/gmot40"),
    protocol=GMOT40_PROTOCOL,
    seq_names=_gmot40_seq_names,
    gt_path=_gmot40_gt_path,
    seq_length=_gmot40_seq_length,
)


def load_gmot40(root: str | Path | None = None, split: str = "test") -> MOTDataset[GtSequence]:
    return load_motchallenge(GMOT40_CONFIG, root=root, split=split)


register_dataset("gmot40")(load_gmot40)
