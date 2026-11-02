from dataclasses import dataclass
from pathlib import Path

from moteval import _moteval
from moteval._rust import call


@dataclass(frozen=True)
class Track:
    """One ``xywh`` pixel box, top-left corner at ``(x, y)``, at one frame of one track.

    ``conf`` is the detection confidence, or for ground truth the gt.txt "consider" flag.
    """

    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    class_id: int = 1


def read_mot(path: Path, *, class_column: bool = False) -> list[Track]:
    """Without ``class_column`` every row gets ``class_id=1``. With it, the 8th column is
    the class id, and a row without one or with a negative one raises `ValueError`.
    """
    return [Track(*row) for row in call(_moteval.read_mot, path, class_column)]


def write_mot(path: Path, tracks: list[Track]) -> None:
    call(_moteval.write_mot, path, tracks)


@dataclass(frozen=True)
class MaskTrack:
    """One mask at one frame of one track; ``rle`` is a pycocotools compressed-RLE counts string."""

    frame: int
    track_id: int
    class_id: int
    img_h: int
    img_w: int
    rle: str


def read_mots(path: Path) -> list[MaskTrack]:
    return [MaskTrack(*row) for row in call(_moteval.read_mots, path)]


def write_mots(path: Path, tracks: list[MaskTrack]) -> None:
    call(_moteval.write_mots, path, tracks)
