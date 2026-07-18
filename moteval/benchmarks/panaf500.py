"""PanAf500: per-video JSON annotations (xyxy boxes), converted to canonical xywh.

Layout: ``<root>/annotations/<split>/<video_id>.json`` -- a JSON object shaped
``{"annotations": [{"frame_id": int, "detections": [{"bbox": [x1,y1,x2,y2],
"ape_id": int, ...}]}]}``. moteval never reads the accompanying
``<root>/frames/<split>/<video_id>/*.jpg`` frames -- evaluation doesn't need
them. Splits are accepted as they exist on disk (train/validation/test); no
remapping. ``frame_id`` is already 1-indexed. This loader never touches
`read_mot`, so it stamps ``class_id`` on every `Track` explicitly rather than
relying on the dataclass default.
"""

import json
from pathlib import Path

from moteval.benchmarks.base import register_dataset
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import Track

PANAF500_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
PANAF500_PROTOCOL = Protocol(
    name="panaf500",
    frame_convention=PANAF500_CONVENTION,
    eval_classes=(1,),
)
_CLASS_ID = 1
_DEFAULT_ROOT = Path("data/benchmarks/panaf500")


def _xyxy_to_xywh(bbox: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def _load_sequence(ann_path: Path) -> GtSequence:
    data = json.loads(ann_path.read_text())
    tracks: list[Track] = []
    for frame_ann in data["annotations"]:
        frame = frame_ann["frame_id"]
        for det in frame_ann["detections"]:
            x, y, w, h = _xyxy_to_xywh(det["bbox"])
            tracks.append(
                Track(
                    frame=frame,
                    track_id=det["ape_id"],
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    conf=1.0,
                    class_id=_CLASS_ID,
                )
            )
    if not data["annotations"]:
        raise ValueError(f"cannot derive sequence length for empty gt: {ann_path.stem!r}")
    num_timesteps = max(a["frame_id"] for a in data["annotations"])
    return GtSequence(name=ann_path.stem, num_timesteps=num_timesteps, tracks=tuple(tracks))


def load_panaf500(
    root: str | Path | None = None, split: str = "validation"
) -> MOTDataset[GtSequence]:
    base = Path(root) if root is not None else _DEFAULT_ROOT
    ann_dir = base / "annotations" / split
    if not ann_dir.is_dir():
        raise ValueError(f"split directory not found: {ann_dir}")
    sequences = tuple(_load_sequence(p) for p in sorted(ann_dir.glob("*.json")))
    return MOTDataset(name="panaf500", split=split, sequences=sequences, protocol=PANAF500_PROTOCOL)


register_dataset("panaf500")(load_panaf500)
