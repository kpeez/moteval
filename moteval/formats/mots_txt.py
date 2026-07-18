"""MOTS Challenge txt format: whitespace-separated ``frame id class img_h img_w rle`` rows.

One row per mask. ``rle`` is a pycocotools compressed-RLE counts string
(KITTI-MOTS / MOTS Challenge style). Class 10 marks ignore regions in GT files;
routing those rows to `MaskGtSequence.ignore_regions` is the loader's job, per
the benchmark's declared protocol — `read_mots` itself returns every row.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaskTrack:
    """One RLE mask at one frame, tagged with the track id it belongs to.

    ``rle`` is the compressed counts string exactly as read from the file; the
    mask's pycocotools dict form is ``{"size": [img_h, img_w], "counts":
    rle.encode()}``. The frame number is interpreted under a declared
    `FrameConvention`.
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
