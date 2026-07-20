"""On-disk annotation formats: MOTChallenge box rows and MOTS mask rows.

Two parallel reader/writer pairs — parallel because the formats are, not because
one derives from the other:

- **Boxes** (`Track`, `read_mot`, `write_mot`): comma-separated
  ``frame,id,x,y,w,h,conf`` rows, one per detection. Boxes are ``xywh`` in
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


@dataclass(frozen=True)
class Track:
    """One box at one frame, tagged with the track id it belongs to.

    For ground-truth rows read from a ``gt.txt`` file, ``conf`` carries the file's
    7th column (the "consider" flag). For predictions it is the detection
    confidence. ``class_id`` defaults to pedestrian (1). `read_mot` parses no class
    column, so every row it returns keeps that default — multi-class GT loaders
    must construct `Track` rows with an explicit ``class_id`` themselves rather
    than reuse `read_mot`.
    """

    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    class_id: int = 1


def read_mot(path: Path) -> list[Track]:
    """Parse a MOTChallenge txt file into `Track` rows."""
    tracks: list[Track] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 6:
            raise ValueError(f"malformed MOT row in {path}:{lineno}: {line!r}")
        try:
            conf = float(fields[6]) if len(fields) > 6 else 1.0
            track = Track(
                frame=int(fields[0]),
                track_id=int(fields[1]),
                x=float(fields[2]),
                y=float(fields[3]),
                w=float(fields[4]),
                h=float(fields[5]),
                conf=conf,
            )
        except ValueError as err:
            raise ValueError(f"malformed MOT row in {path}:{lineno}: {line!r}") from err
        tracks.append(track)
    return tracks


def write_mot(path: Path, tracks: list[Track]) -> None:
    """Write `Track` rows as MOTChallenge predictions (trailing fields = -1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(tracks, key=lambda t: (t.frame, t.track_id))
    lines = [
        f"{t.frame},{t.track_id},{t.x:.2f},{t.y:.2f},{t.w:.2f},{t.h:.2f},{t.conf:.4f},-1,-1,-1"
        for t in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


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
    tracks: list[MaskTrack] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"malformed MOTS row in {path}:{lineno}: {line!r}")
        try:
            track = MaskTrack(
                frame=int(fields[0]),
                track_id=int(fields[1]),
                class_id=int(fields[2]),
                img_h=int(fields[3]),
                img_w=int(fields[4]),
                rle=fields[5],
            )
        except ValueError as err:
            raise ValueError(f"malformed MOTS row in {path}:{lineno}: {line!r}") from err
        tracks.append(track)
    return tracks


def write_mots(path: Path, tracks: list[MaskTrack]) -> None:
    """Write `MaskTrack` rows as a MOTS txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(tracks, key=lambda t: (t.frame, t.track_id))
    lines = [f"{t.frame} {t.track_id} {t.class_id} {t.img_h} {t.img_w} {t.rle}" for t in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
