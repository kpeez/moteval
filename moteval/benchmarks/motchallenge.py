"""Generic MOTChallenge-directory adapter, configured per benchmark.

Layout: ``<root>/<split>/<seq>/gt/gt.txt`` plus ``<root>/<split>/<seq>/seqinfo.ini``
(``[Sequence]`` section, ``seqLength=N``). Sequence discovery is the split
directory's subdirectories, sorted. GT rows are the standard 7-field MOTChallenge
prefix (``frame,id,x,y,w,h,conf``); `read_mot` parses no class column, so every
`Track` it returns silently gets ``class_id=1``. That is coincidentally correct
for single-class benchmarks (DanceTrack, SportsMOT: pedestrian == class 1) but
would be WRONG for multi-class GT -- a future multi-class benchmark must not
reuse this adapter's GT loading naively. `MOTChallengeConfig.class_id` is the
adapter's only class-assignment hook today; a class-column parser can be added
as a config field later (e.g. a row-parsing callable) without breaking DanceTrack
or SportsMOT, which simply leave it at the default.
"""

import configparser
from dataclasses import dataclass, replace
from pathlib import Path

from moteval.data.model import GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import read_mot

_SEQINFO_SECTION = "Sequence"
_SEQLENGTH_KEY = "seqLength"


@dataclass(frozen=True)
class MOTChallengeConfig:
    """Per-benchmark configuration for the generic MOTChallenge adapter.

    ``class_id`` is the class every GT `Track` is tagged with, since `read_mot`
    parses no class column itself. Single-class benchmarks leave it at the
    pedestrian default (1).
    """

    name: str
    default_root: Path
    protocol: Protocol
    class_id: int = 1


def _read_seq_length(seq_dir: Path) -> int:
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.is_file():
        raise ValueError(f"missing seqinfo.ini for sequence {seq_dir.name!r} at {seqinfo_path}")
    parser = configparser.ConfigParser()
    try:
        parser.read(seqinfo_path)
    except configparser.Error as err:
        raise ValueError(
            f"malformed seqinfo.ini for sequence {seq_dir.name!r} at {seqinfo_path}: {err}"
        ) from err
    if not parser.has_option(_SEQINFO_SECTION, _SEQLENGTH_KEY):
        raise ValueError(
            f"malformed seqinfo.ini for sequence {seq_dir.name!r} at {seqinfo_path}: "
            f"missing [{_SEQINFO_SECTION}] {_SEQLENGTH_KEY}"
        )
    try:
        return parser.getint(_SEQINFO_SECTION, _SEQLENGTH_KEY)
    except ValueError as err:
        raise ValueError(
            f"malformed seqinfo.ini for sequence {seq_dir.name!r} at {seqinfo_path}: "
            f"{_SEQLENGTH_KEY} is not an integer"
        ) from err


def _load_sequence(seq_dir: Path, class_id: int) -> GtSequence:
    num_timesteps = _read_seq_length(seq_dir)
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise ValueError(f"missing gt.txt for sequence {seq_dir.name!r} at {gt_path}")
    tracks = tuple(replace(t, class_id=class_id) for t in read_mot(gt_path))
    return GtSequence(name=seq_dir.name, num_timesteps=num_timesteps, tracks=tracks)


def load_motchallenge(
    config: MOTChallengeConfig, root: str | Path | None = None, split: str = "val"
) -> MOTDataset:
    """Load a MOTChallenge-layout split into a canonical `MOTDataset`."""
    base = Path(root) if root is not None else config.default_root
    split_dir = base / split
    if not split_dir.is_dir():
        raise ValueError(f"split directory not found: {split_dir}")
    seq_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    sequences = tuple(_load_sequence(seq_dir, config.class_id) for seq_dir in seq_dirs)
    return MOTDataset(name=config.name, split=split, sequences=sequences, protocol=config.protocol)
