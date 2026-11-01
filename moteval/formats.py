"""On-disk annotation formats: MOTChallenge box rows and MOTS mask rows.

Two parallel reader/writer pairs — parallel because the formats are, not because
one derives from the other:

- **Boxes** (`Track`, `read_mot`, `write_mot`): comma-separated
  ``frame,id,x,y,w,h,conf[,class,...]`` rows, one per detection. Boxes are ``xywh`` in
  pixels with the top-left corner at ``(x, y)``.
- **Masks** (`MaskTrack`, `read_mots`, `write_mots`): whitespace-separated
  ``frame id class img_h img_w rle`` rows, one per mask. ``rle`` is a
  pycocotools compressed-RLE counts string (KITTI-MOTS / MOTS Challenge style).

These are the lingua franca between predictions, ground truth, and the metrics:
everything read from disk is a list of `Track` or `MaskTrack`, and frame
numbers are interpreted under a declared `FrameConvention`.
"""

from dataclasses import dataclass
from pathlib import Path

from moteval import _moteval
from moteval._rust import call


@dataclass(frozen=True)
class Track:
    """One box at one frame, tagged with the track id it belongs to.

    For ground-truth rows read from a ``gt.txt`` file, ``conf`` carries the file's
    7th column (the "consider" flag). For predictions it is the detection
    confidence. ``class_id`` defaults to pedestrian (1). `read_mot` keeps that
    default unless the caller asks it to read the class column (``class_column=True``).
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
    """Parse a MOTChallenge txt file into `Track` rows.

    With ``class_column=False`` (the default) the 8th column is ignored and every
    row gets ``class_id=1``: many single-class prediction files hold ``-1`` there.
    With ``class_column=True`` the 8th column is the class id. A row without it, or
    with a negative class (``-1`` is the MOTChallenge "unset" marker), raises rather
    than silently scoring in no class. `evaluate` sets it for multi-class protocols
    only.
    """
    return [Track(*row) for row in call(_moteval.read_mot, path, class_column)]


def write_mot(path: Path, tracks: list[Track]) -> None:
    """Write `Track` rows as MOTChallenge predictions.

    Column 8 holds ``class_id``, so `read_mot` with ``class_column=True`` reads the
    classes back. The two trailing fields are ``-1``.
    """
    call(_moteval.write_mot, path, tracks)


@dataclass(frozen=True)
class MaskTrack:
    """One RLE mask at one frame, tagged with the track id it belongs to.

    ``rle`` is the compressed counts string exactly as read from the file; the
    mask's pycocotools dict form is ``{"size": [img_h, img_w], "counts":
    rle.encode()}``. Class 10 marks ignore regions in GT files; routing those
    rows to `MaskGtSequence.ignore_regions` is the loader's job, per the
    benchmark's declared protocol — `read_mots` itself returns every row.
    """

    frame: int
    track_id: int
    class_id: int
    img_h: int
    img_w: int
    rle: str


def read_mots(path: Path) -> list[MaskTrack]:
    """Parse a MOTS txt file into `MaskTrack` rows."""
    return [MaskTrack(*row) for row in call(_moteval.read_mots, path)]


def write_mots(path: Path, tracks: list[MaskTrack]) -> None:
    """Write `MaskTrack` rows as a MOTS txt file, sorted by ``(frame, track_id)``."""
    call(_moteval.write_mots, path, tracks)
