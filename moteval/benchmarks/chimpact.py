"""ChimpACT: COCO-JSON keyframe annotations sampled every 10th frame, natively
0-indexed.

Layout: ``<root>/ChimpACT_release_v1/labels/<clip>.json`` -- one COCO-style JSON
file per clip with ``images`` and ``annotations`` keys (moteval never reads the
accompanying ``videos_full/<clip>.mp4`` -- evaluation doesn't need it). Each
``images`` entry's ``file_name`` stem is a keyframe *block index*, not a video
frame number: block N was sampled at video frame N*10, and those video frame
numbers are 0-indexed on disk, matching this loader's declared
``FrameConvention(first_frame=0)`` rather than shifting to 1-indexed like the
legacy track-zoo loader did (ADR-0002: frame-indexing is a declared loader
parameter). Split membership has no on-disk marker in the flat ``labels/``
directory, so ``_VAL_CLIPS``/``_TEST_CLIPS`` (copied verbatim from the official
``tools/create_coco_format.py``, ShirleyMaxx/ChimpACT@master) are the split
partition; anything else falls into ``train``.

Interpolation/hold semantics replicate legacy track-zoo exactly (issue #12
decision, logged on the issue): the spec's Goal requires ChimpACT numbers to
stay comparable to arXiv:2511.02591's published baselines, which were
computed with the legacy loader, so where a "cleaner" rule would diverge from
legacy output, legacy wins. Reproducing the official create_coco_format.py /
create_mot_reid_dataset.py pipeline: for a track present in both of two
consecutive keyframe blocks, the 9 interior frames are linearly interpolated
between its two keyframe boxes; for a track with no match in the next
keyframe block (dies mid-clip, or the next block doesn't exist at all), its
last keyframe box is held constant through that one block's 9 interior
frames -- it does not extrapolate further than that, and does not
retroactively reappear if a later, unrelated block exists. ``bbox_id == 23``
is the official MOT converter's unnamed catch-all track and is dropped
wherever it appears.

The JSON carries no explicit video-length field -- its keys are ``info``,
``licenses``, ``categories``, ``images``, ``annotations``, with no ``videos``
entry -- so, consistent with BFT/AnimalTrack/GMOT-40, ``num_timesteps`` is
derived from the last keyframe block. Unlike those loaders' `+1` (stopping
exactly at the last annotated frame), this is `(max_block + 1) * 10`: room
for the last keyframe's own 9-frame hold-tail is required for the hold
behavior above to ever produce output there, and it matches every
round-length real clip exactly (e.g. `clip_9000_10000`'s 100 keyframe blocks
derive `num_timesteps=1000`, its true length). For a non-round-length real
clip (e.g. `clip_0_696`, true length 696 frames, 70 keyframe blocks 0-69),
this derives `num_timesteps=700`, four frames past the clip's real end --
the last keyframe's held box gets extended into 4 frames that never
existed, up to 9 in the worst case. Harmless for the fixture tests here
(there is no real clip to overshoot); must be revisited if ChimpACT
real-data parity is ever added (the final parity gate is issue #20).
"""

import json
from pathlib import Path

from moteval.benchmarks.base import register_dataset
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import Track

_KEYFRAME_STRIDE = 10
_UNNAMED_BBOX_ID = 23
# ChimpACT's single foreground class ("Chimpanzee"). Predictions are read back
# through the standard MOTChallenge txt format (`read_mot`), which parses no
# class column and always yields `class_id=1` -- so this must be 1 for any
# prediction to ever survive the class filter, regardless of the raw COCO
# `category_id` (0) each annotation carries; that field is intentionally never
# read.
_CLASS_ID = 1
_DEFAULT_ROOT = Path("data/benchmarks/chimpact")

CHIMPACT_CONVENTION = FrameConvention(name="0-indexed", first_frame=0)
CHIMPACT_PROTOCOL = Protocol(
    name="chimpact",
    frame_convention=CHIMPACT_CONVENTION,
    eval_classes=(_CLASS_ID,),
)

# Copied verbatim from the official tools/create_coco_format.py
# (ShirleyMaxx/ChimpACT@master) -- proves the 127/17/19 split without
# hardcoding the 127 train clip names.
_VAL_CLIPS = (
    "Azibo_ObsChimp_2015_11_25_d_clip_23000_24000",
    "Azibo_ObsChimp_2015_11_26_a_clip_1000_2000",
    "Azibo_ObsChimp_2016_08_02_c_clip_32000_33000",
    "Azibo_ObsChimp_2017_02_27_a_clip_13000_14000",
    "Azibo_ObsChimp_2017_11_10_clip_7000_8000",
    "Azibo_ObsChimp_2017_11_10_clip_8000_9000",
    "Azibo_ObsChimp_2017_06_22_c_clip_46000_47000",
    "Azibo_ObsChimp_2017_06_22_c_clip_67000_68000",
    "Azibo_ObsChimp_2018_07_11_c_clip_0_1000",
    "Azibo_ObsChimp_2018_07_11_c_clip_1000_2000",
    "Azibo_ObsChimp_2018_07_11_c_clip_3000_4000",
    "Azibo_ObsChimp_2018_07_11_c_clip_6000_7000",
    "Azibo_ObsChimp_2018_07_11_c_clip_17000_18000",
    "Azibo_ObsChimp_2018_07_11_c_clip_18000_19000",
    "Azibo_ObsChimp_2018_08_06_a_clip_7000_8000",
    "Azibo_ObsNatascha_2018_06_29_a_clip_15000_16000",
    "Azibo_ObsNatascha_2018_06_29_a_clip_16000_17000",
)
_TEST_CLIPS = (
    "Azibo_ObsChimp_2015_11_25_d_clip_1000_2000",
    "Azibo_ObsChimp_2015_11_26_a_clip_0_1000",
    "Azibo_ObsChimp_2015_11_26_a_clip_2000_3000",
    "Azibo_ObsChimp_2016_08_02_c_clip_33000_34000",
    "Azibo_ObsChimp_2016_08_15_b_clip_2000_3000",
    "Azibo_ObsChimp_2016_10_27_c_clip_0_1000",
    "Azibo_ObsChimp_2017_02_27_a_clip_14000_15000",
    "Azibo_ObsChimp_2017_11_10_clip_6000_7000",
    "Azibo_ObsChimp_2017_06_22_c_clip_44000_45000",
    "Azibo_ObsChimp_2017_06_22_c_clip_68000_69000",
    "Azibo_ObsChimp_2018_07_06_d_clip_0_696",
    "Azibo_ObsChimp_2018_07_11_c_clip_2000_3000",
    "Azibo_ObsChimp_2018_07_11_c_clip_8000_9000",
    "Azibo_ObsChimp_2018_07_11_c_clip_16000_17000",
    "Azibo_ObsChimp_2018_07_11_c_clip_19000_20000",
    "Azibo_ObsChimp_2018_08_06_a_clip_6000_7000",
    "Azibo_ObsChimp_2018_08_06_a_clip_8000_9000",
    "Azibo_ObsNatascha_2018_06_29_a_clip_14000_15000",
    "Azibo_ObsNatascha_2018_06_29_a_clip_17000_17712",
)


def _keyframe_block(image: dict) -> int:
    return int(Path(image["file_name"]).stem)


def _keyframe_blocks(labels: dict) -> dict[int, list[dict]]:
    image_id_to_block = {img["id"]: _keyframe_block(img) for img in labels["images"]}
    blocks: dict[int, list[dict]] = {}
    for ann in labels["annotations"]:
        if ann["bbox_id"] == _UNNAMED_BBOX_ID:
            continue
        blocks.setdefault(image_id_to_block[ann["image_id"]], []).append(ann)
    return blocks


def _interpolated_box(
    cur_bbox: list[float], next_bbox: list[float], t: float
) -> tuple[float, float, float, float]:
    x, y, w, h = (c * (1 - t) + n * t for c, n in zip(cur_bbox, next_bbox, strict=True))
    return x, y, w, h


def _clip_tracks(labels: dict) -> tuple[Track, ...]:
    blocks = _keyframe_blocks(labels)
    tracks: list[Track] = []
    for block in sorted(blocks):
        objs = blocks[block]
        frame = block * _KEYFRAME_STRIDE
        for obj in objs:
            x, y, w, h = obj["bbox"]
            tracks.append(
                Track(
                    frame=frame,
                    track_id=obj["bbox_id"],
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    conf=1.0,
                    class_id=_CLASS_ID,
                )
            )
        next_by_id = {o["bbox_id"]: o for o in blocks.get(block + 1, ())}
        for obj in objs:
            match = next_by_id.get(obj["bbox_id"])
            for offset in range(1, _KEYFRAME_STRIDE):
                if match is not None:
                    x, y, w, h = _interpolated_box(
                        obj["bbox"], match["bbox"], offset / _KEYFRAME_STRIDE
                    )
                else:
                    # No match in the next block (track dies here, or there is
                    # no next block at all): hold this keyframe's box constant
                    # rather than extrapolating any further, matching legacy.
                    x, y, w, h = obj["bbox"]
                tracks.append(
                    Track(
                        frame=frame + offset,
                        track_id=obj["bbox_id"],
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        conf=1.0,
                        class_id=_CLASS_ID,
                    )
                )
    return tuple(tracks)


def _num_timesteps(labels: dict, clip_name: str) -> int:
    if not labels["images"]:
        raise ValueError(f"cannot derive sequence length for empty gt: {clip_name!r}")
    max_block = max(_keyframe_block(img) for img in labels["images"])
    return (max_block + 1) * _KEYFRAME_STRIDE


def _load_clip(label_path: Path) -> GtSequence:
    labels = json.loads(label_path.read_text())
    tracks = _clip_tracks(labels)
    num_timesteps = _num_timesteps(labels, label_path.stem)
    return GtSequence(name=label_path.stem, num_timesteps=num_timesteps, tracks=tracks)


def _split_clip_names(label_dir: Path, split: str) -> list[str]:
    names = sorted(p.stem for p in label_dir.glob("*.json"))
    if split == "val":
        return [name for name in names if name in _VAL_CLIPS]
    if split == "test":
        return [name for name in names if name in _TEST_CLIPS]
    if split != "train":
        raise ValueError(f"unknown chimpact split {split!r}; expected 'train', 'val', or 'test'")
    return [name for name in names if name not in _VAL_CLIPS and name not in _TEST_CLIPS]


def load_chimpact(root: str | Path | None = None, split: str = "val") -> MOTDataset[GtSequence]:
    base = Path(root) if root is not None else _DEFAULT_ROOT
    label_dir = base / "ChimpACT_release_v1" / "labels"
    if not label_dir.is_dir():
        raise ValueError(f"labels directory not found: {label_dir}")
    names = _split_clip_names(label_dir, split)
    sequences = tuple(_load_clip(label_dir / f"{name}.json") for name in names)
    return MOTDataset(name="chimpact", split=split, sequences=sequences, protocol=CHIMPACT_PROTOCOL)


register_dataset("chimpact")(load_chimpact)
