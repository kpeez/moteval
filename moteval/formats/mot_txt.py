"""MOTChallenge box format: read and write ``frame,id,x,y,w,h,conf`` rows.

One row per detection. Boxes are ``xywh`` in pixels with the top-left corner at
``(x, y)``. This is the lingua franca between predictions, ground truth, and the
metrics: everything read from disk is a list of `Track`.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Track:
    """One box at one frame, tagged with the track id it belongs to.

    For ground-truth rows read from a ``gt.txt`` file, ``conf`` carries the file's
    7th column (the "consider" flag). For predictions it is the detection
    confidence. The frame number is interpreted under a declared `FrameConvention`.
    """

    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float


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
