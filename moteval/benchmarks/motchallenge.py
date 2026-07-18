"""Generic MOTChallenge-directory adapter, configured per benchmark.

Default layout: ``<root>/<split>/<seq>/gt/gt.txt`` plus
``<root>/<split>/<seq>/seqinfo.ini`` (``[Sequence]`` section, ``seqLength=N``).
Sequence discovery is the split directory's subdirectories, sorted. GT rows are
the standard 7-field MOTChallenge prefix (``frame,id,x,y,w,h,conf``); `read_mot`
parses no class column, so every `Track` it returns silently gets ``class_id=1``.
That is coincidentally correct for single-class benchmarks (DanceTrack,
SportsMOT: pedestrian == class 1) but would be WRONG for multi-class GT -- a
future multi-class benchmark must not reuse this adapter's GT loading naively.
`MOTChallengeConfig.class_id` is the adapter's only class-assignment hook today;
a class-column parser can be added as a config field later (e.g. a row-parsing
callable) without breaking DanceTrack or SportsMOT, which simply leave it at
the default.

Some benchmarks (BFT, AnimalTrack, GMOT-40) keep MOTChallenge-style GT rows but
deviate from that default layout -- no ``seqinfo.ini``, GT living in a flat
per-split directory instead of nested under each sequence, or split membership
coming from an external list file. `MOTChallengeConfig.seq_names`, ``gt_path``,
``seq_length``, and ``ignore_path`` are the minimal hooks that let those
benchmarks reuse this adapter instead of forking it; each defaults to the
standard-layout behavior (``ignore_path`` defaults to no ignore file, i.e. an
empty ``GtSequence.ignore_regions``).
"""

import configparser
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from moteval.data.model import GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import Track, read_mot

_SEQINFO_SECTION = "Sequence"
_SEQLENGTH_KEY = "seqLength"

SeqNamesFn = Callable[[Path, str], list[str]]
GtPathFn = Callable[[Path, str, str], Path]
SeqLengthFn = Callable[[Path, str, str, tuple[Track, ...]], int]
IgnorePathFn = Callable[[Path, str, str], Path | None]


def _default_seq_names(base: Path, split: str) -> list[str]:
    split_dir = base / split
    if not split_dir.is_dir():
        raise ValueError(f"split directory not found: {split_dir}")
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def _default_gt_path(base: Path, split: str, seq_name: str) -> Path:
    return base / split / seq_name / "gt" / "gt.txt"


def _seqinfo_seq_length(base: Path, split: str, seq_name: str, tracks: tuple[Track, ...]) -> int:
    return _read_seq_length(base / split / seq_name)


def _default_ignore_path(base: Path, split: str, seq_name: str) -> Path | None:
    return None


def max_frame_seq_length(seq_name: str, tracks: tuple[Track, ...], offset: int = 0) -> int:
    """Derive ``num_timesteps`` from the last annotated frame, for benchmarks with
    no ``seqinfo.ini`` source (BFT, AnimalTrack, GMOT-40). ``offset`` is 1 for
    0-indexed frame conventions, where the last frame *index* needs +1 to become a
    count. Raises loudly on empty GT since there is no other frame-count source to
    fall back on.
    """
    if not tracks:
        raise ValueError(f"cannot derive sequence length for empty gt: {seq_name!r}")
    return max(t.frame for t in tracks) + offset


@dataclass(frozen=True)
class MOTChallengeConfig:
    """Per-benchmark configuration for the generic MOTChallenge adapter.

    ``class_id`` is the class every GT `Track` is tagged with, since `read_mot`
    parses no class column itself. Single-class benchmarks leave it at the
    pedestrian default (1). ``seq_names``, ``gt_path``, and ``seq_length`` default
    to the standard ``<root>/<split>/<seq>/gt/gt.txt`` + ``seqinfo.ini`` layout;
    override them for benchmarks whose real distribution deviates from it.
    ``ignore_path`` returns the crowd-ignore-region file for a sequence, or
    ``None`` when a benchmark has no such file; it defaults to always returning
    ``None`` so benchmarks without ignore regions are unaffected.
    """

    name: str
    default_root: Path
    protocol: Protocol
    class_id: int = 1
    seq_names: SeqNamesFn = _default_seq_names
    gt_path: GtPathFn = _default_gt_path
    seq_length: SeqLengthFn = _seqinfo_seq_length
    ignore_path: IgnorePathFn = _default_ignore_path


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


def _load_sequence(base: Path, split: str, seq_name: str, config: MOTChallengeConfig) -> GtSequence:
    gt_path = config.gt_path(base, split, seq_name)
    if not gt_path.is_file():
        raise ValueError(f"missing gt.txt for sequence {seq_name!r} at {gt_path}")
    tracks = tuple(replace(t, class_id=config.class_id) for t in read_mot(gt_path))
    num_timesteps = config.seq_length(base, split, seq_name, tracks)
    ignore_path = config.ignore_path(base, split, seq_name)
    ignore_regions = (
        tuple(read_mot(ignore_path)) if ignore_path is not None and ignore_path.is_file() else ()
    )
    return GtSequence(
        name=seq_name, num_timesteps=num_timesteps, tracks=tracks, ignore_regions=ignore_regions
    )


def load_motchallenge(
    config: MOTChallengeConfig, root: str | Path | None = None, split: str = "val"
) -> MOTDataset:
    """Load a MOTChallenge-layout split into a canonical `MOTDataset`."""
    base = Path(root) if root is not None else config.default_root
    seq_names = config.seq_names(base, split)
    sequences = tuple(_load_sequence(base, split, name, config) for name in seq_names)
    return MOTDataset(name=config.name, split=split, sequences=sequences, protocol=config.protocol)
