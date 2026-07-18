"""DanceTrack: MOTChallenge-layout, single-class (pedestrian), 1-indexed."""

from pathlib import Path

from moteval.benchmarks.base import register_dataset
from moteval.benchmarks.motchallenge import MOTChallengeConfig, load_motchallenge
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol

DANCETRACK_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
DANCETRACK_PROTOCOL = Protocol(
    name="dancetrack",
    frame_convention=DANCETRACK_CONVENTION,
    eval_classes=(1,),
)
DANCETRACK_CONFIG = MOTChallengeConfig(
    name="dancetrack",
    default_root=Path("data/benchmarks/dancetrack"),
    protocol=DANCETRACK_PROTOCOL,
)


def load_dancetrack(root: str | Path | None = None, split: str = "val") -> MOTDataset[GtSequence]:
    return load_motchallenge(DANCETRACK_CONFIG, root=root, split=split)


register_dataset("dancetrack")(load_dancetrack)
