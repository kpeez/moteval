"""MOTS Challenge (MOTS20): mask GT, pedestrian class 2, class-10 ignore, 1-indexed.

Layout: ``<root>/<split>/<seq>/gt/gt.txt`` (whitespace-separated MOTS rows) plus
``seqinfo.ini`` for sequence length, mirroring the MOTChallenge directory
convention MOTS20 ships with. Class-10 GT rows are routed to
``MaskGtSequence.ignore_regions``; every other class is kept as a plain track
row and the protocol's class filter selects pedestrians (class 2) at eval time.

``matching_fill=-10000`` replicates upstream's MOTS preprocessing exactly: its
Hungarian matching fills below-threshold pairs with -10000 where the box path
uses 0, which can tie-break differently. MOTS GT has no confidence column, so
``drop_zero_conf_gt=False`` (loaders fill conf=1); GT is never filtered beyond
the class selection, exactly as upstream MOTS never drops GT.
"""

from pathlib import Path

from moteval.benchmarks.motchallenge import _read_seq_length
from moteval.data.model import FrameConvention, MaskGtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.data.registry import register_dataset
from moteval.formats import read_mots

MOTS20_IGNORE_CLASS = 10
MOTS20_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
MOTS20_PROTOCOL = Protocol(
    name="mots20",
    frame_convention=MOTS20_CONVENTION,
    eval_classes=(2,),
    drop_zero_conf_gt=False,
    matching_fill=-10000.0,
)
MOTS20_DEFAULT_ROOT = Path("data/benchmarks/mots20")


def _load_sequence(base: Path, split: str, seq_name: str) -> MaskGtSequence:
    gt_path = base / split / seq_name / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise ValueError(f"missing gt.txt for sequence {seq_name!r} at {gt_path}")
    rows = read_mots(gt_path)
    tracks = tuple(t for t in rows if t.class_id != MOTS20_IGNORE_CLASS)
    ignore_regions = tuple(t for t in rows if t.class_id == MOTS20_IGNORE_CLASS)
    num_timesteps = _read_seq_length(base / split / seq_name)
    return MaskGtSequence(
        name=seq_name, num_timesteps=num_timesteps, tracks=tracks, ignore_regions=ignore_regions
    )


def load_mots20(root: str | Path | None = None, split: str = "train") -> MOTDataset[MaskGtSequence]:
    base = Path(root) if root is not None else MOTS20_DEFAULT_ROOT
    split_dir = base / split
    if not split_dir.is_dir():
        raise ValueError(f"split directory not found: {split_dir}")
    seq_names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    sequences = tuple(_load_sequence(base, split, name) for name in seq_names)
    return MOTDataset(name="mots20", split=split, sequences=sequences, protocol=MOTS20_PROTOCOL)


register_dataset("mots20")(load_mots20)
