"""SportsMOT: MOTChallenge-layout, single-class (player), 1-indexed."""

from pathlib import Path

from moteval.benchmarks.motchallenge import MOTChallengeConfig, load_motchallenge
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.data.registry import register_dataset

SPORTSMOT_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
SPORTSMOT_PROTOCOL = Protocol(
    name="sportsmot",
    frame_convention=SPORTSMOT_CONVENTION,
    eval_classes=(1,),
)
SPORTSMOT_CONFIG = MOTChallengeConfig(
    name="sportsmot",
    default_root=Path("data/benchmarks/sportsmot"),
    protocol=SPORTSMOT_PROTOCOL,
)


def load_sportsmot(root: str | Path | None = None, split: str = "val") -> MOTDataset[GtSequence]:
    return load_motchallenge(SPORTSMOT_CONFIG, root=root, split=split)


register_dataset("sportsmot")(load_sportsmot)
