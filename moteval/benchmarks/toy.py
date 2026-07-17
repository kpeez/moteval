"""Bundled toy benchmark: two tiny 1-indexed MOTChallenge sequences, in memory.

Two tracks per sequence over five frames. Ground truth is generated rather than
read from disk so the tracer bullet stays hermetic; predictions are read from a
`<seq>.txt` directory by `evaluate`.
"""

from moteval.benchmarks.base import register_dataset
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.formats.mot_txt import Track

TOY_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)


def _linear_track(
    track_id: int, x0: float, y0: float, dx: float, w: float, h: float
) -> list[Track]:
    return [
        Track(frame=f, track_id=track_id, x=x0 + dx * (f - 1), y=y0, w=w, h=h, conf=1.0)
        for f in range(1, 6)
    ]


@register_dataset("toy")
def load_toy() -> MOTDataset:
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
        frame_convention=TOY_CONVENTION,
    )
