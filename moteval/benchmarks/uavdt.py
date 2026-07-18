"""UAVDT: MOTChallenge-style GT rows, no seqinfo.ini, GT and ignore-region files
living in a flat directory keyed by sequence name.

Layout: ``<root>/UAV-benchmark-MOTD_v1.0/GT/<seq>_gt.txt`` (GT), with a sibling
``<seq>_gt_ignore.txt`` per sequence declaring crowd-ignore regions. moteval
never reads the accompanying ``<root>/UAV-benchmark-M/<seq>/img%06d.jpg``
frames -- evaluation doesn't need them. No train/test split ships with the
data; "all" exposes every sequence with a ``*_gt.txt`` file (also excludes the
sibling ``*_gt_ignore.txt``/``*_gt_whole.txt`` files, since neither ends in
exactly ``_gt.txt``).

GT columns are ``frame,id,x,y,w,h,score,in-view,occlusion`` per the MOTD
README; `read_mot` parses the standard 7-field MOTChallenge prefix, so
``score`` (the "consider" flag: 1 evaluates the box, 0 ignores it) lands in
`Track.conf` unfiltered -- scoring, not the loader, drops conf-zero rows (see
`Protocol.drop_zero_conf_gt`). Frames are 1-indexed, matching `Track.frame`, so
no shift. Sequence length has no seqinfo.ini source; it is derived from the
last annotated frame (see the same note in `bft.py`).

The legacy track-zoo loader parsed but never applied ignore regions -- a
protocol gap, not a decision (CLAUDE.md). Here, `<seq>_gt_ignore.txt` rows
(same 9-column format, id/score/in-view/occlusion irrelevant per
`GtSequence.ignore_regions`) populate `GtSequence.ignore_regions` via
`MOTChallengeConfig.ignore_path`; the shared preprocessing engine
(`preprocess_frame`) drops unmatched predictions whose IoA with an ignore
region exceeds `Protocol.ignore_iou_threshold`, never the loader. A sequence
with no ignore file gets an empty `ignore_regions` tuple -- the legacy layout
has no marker for "deliberately no ignore regions" vs. "file just doesn't
exist", so absence is treated as no regions to honor.
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

UAVDT_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
UAVDT_PROTOCOL = Protocol(
    name="uavdt",
    frame_convention=UAVDT_CONVENTION,
    eval_classes=(1,),
)


def _uavdt_gt_dir(base: Path) -> Path:
    return base / "UAV-benchmark-MOTD_v1.0" / "GT"


def _uavdt_seq_names(base: Path, split: str) -> list[str]:
    if split != "all":
        raise ValueError(f"unknown uavdt split {split!r}; expected 'all'")
    gt_dir = _uavdt_gt_dir(base)
    if not gt_dir.is_dir():
        raise ValueError(f"split directory not found: {gt_dir}")
    return sorted(p.name.removesuffix("_gt.txt") for p in gt_dir.glob("*_gt.txt"))


def _uavdt_gt_path(base: Path, split: str, seq_name: str) -> Path:
    return _uavdt_gt_dir(base) / f"{seq_name}_gt.txt"


def _uavdt_ignore_path(base: Path, split: str, seq_name: str) -> Path | None:
    return _uavdt_gt_dir(base) / f"{seq_name}_gt_ignore.txt"


def _uavdt_seq_length(base: Path, split: str, seq_name: str, tracks: tuple[Track, ...]) -> int:
    return max_frame_seq_length(seq_name, tracks)


UAVDT_CONFIG = MOTChallengeConfig(
    name="uavdt",
    default_root=Path("data/benchmarks/uavdt"),
    protocol=UAVDT_PROTOCOL,
    seq_names=_uavdt_seq_names,
    gt_path=_uavdt_gt_path,
    seq_length=_uavdt_seq_length,
    ignore_path=_uavdt_ignore_path,
)


def load_uavdt(root: str | Path | None = None, split: str = "all") -> MOTDataset[GtSequence]:
    return load_motchallenge(UAVDT_CONFIG, root=root, split=split)


register_dataset("uavdt")(load_uavdt)
