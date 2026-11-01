"""Dump pycocotools, boundary-match, and J&F results on seeded masks for the Rust port.

Writes four files under ``rust/testdata/``, each checked bit for bit by a Rust test that
regenerates the same masks from the seeds (``maskgen.py`` / ``tests/maskgen/mod.rs``):

- ``rle.json`` (``tests/rle.rs``): 1,000 masks in 250 same-size families. Per mask:
  ``encode``, ``decode``, ``area``, ``toBbox``, and the string round trip through
  ``merge([rle])``. Per family: ``iou`` with every ``iscrowd`` false, every one true, and
  a mixed split; ``merge`` union and intersection of the family and of its first pair;
  moteval's ``masks_overlap``. Across neighbouring families (usually other sizes):
  ``iou`` and ``merge``. Plus non-canonical counts strings that parse like canonical ones,
  and an overrun string that ``decode`` rejects. ``rle_iou.json`` holds the ``iou``
  matrices, apart so that each file stays below the repository's 500 KB file limit.
- ``jf_boundary.json`` (``tests/jf.rs``): 1,000 mask pairs. ``moteval.metrics.jf.
  _boundary_points`` of each mask, and scipy ``KDTree`` match counts within a radius.
- ``jf_sequences.json`` (``tests/jf.rs``): 300 random J&F sequences scored by Python
  moteval's ``JAndF`` (scores or the error it raises), and its three combiners over
  groups of the scored sequences.

Run with the reference interpreter (CPython 3.14.8, numpy 2.5.1, scipy 1.18.0,
pycocotools 2.0.11):

    artifacts/rust-port/ref-venv/bin/python rust/scripts/dump_masks.py
"""

import json
import sys
import warnings
from pathlib import Path
from typing import cast

import numpy as np
from pycocotools import mask as mask_utils
from scipy.spatial import KDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
# After site-packages: `moteval` is the pure-Python reference (rust/scripts/reference.py).
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from rust.scripts.reference import require_reference

require_reference()

from moteval.data.similarity import masks_overlap  # pyrefly: ignore[missing-import]

from moteval.data.model import MaskGeometry, RleMask, SequenceData
from moteval.metrics.jf import JAndF, _boundary_points  # pyrefly: ignore[missing-module-attribute]
from rust.scripts.maskgen import (
    FAMILIES,
    FAMILY_SIZE,
    boundary_case,
    family_seed,
    fnv1a64_bytes,
    fnv1a64_ints,
    jf_sequence,
    mask_size,
    random_mask,
)
from rust.scripts.testgen import SplitMix64, f64_bits

OUT_DIR = Path(__file__).resolve().parents[1] / "testdata"
JF_SEQUENCES = 300
BOUNDARY_PAIRS = 1000
# Combiner groups over the scored J&F sequences: sizes cycle through these.
GROUP_SIZES = (1, 2, 3, 5, 8)


def bits(values) -> list[int]:
    return [f64_bits(float(x)) for x in np.asarray(values, dtype=np.float64).ravel()]


def to_rle(m: bytearray, h: int, w: int) -> RleMask:
    arr = np.frombuffer(bytes(m), dtype=np.uint8).reshape((h, w), order="F")
    rle = cast(RleMask, mask_utils.encode(np.asfortranarray(arr)))
    decoded = mask_utils.decode(rle)
    assert decoded.shape == (h, w) and np.array_equal(decoded, arr), "decode round trip"
    return rle


def merge(rles: list[RleMask], intersect: bool = False) -> RleMask:
    return cast(RleMask, mask_utils.merge(rles, intersect=intersect))


def text(rle: RleMask) -> str:
    """The counts string as text (pycocotools returns ``bytes``)."""
    counts = rle["counts"]
    return counts.decode() if isinstance(counts, bytes) else counts


def rle_json(rle: RleMask) -> dict:
    return {"h": int(rle["size"][0]), "w": int(rle["size"][1]), "counts": text(rle)}


def iou_bits(dt: list[RleMask], gt: list[RleMask], crowd: list[int]) -> list[int]:
    ious = mask_utils.iou(dt, gt, crowd)
    return bits(np.asarray(ious).reshape(len(dt), len(gt)))


def pad_groups(counts: bytes, rng: SplitMix64) -> bytes:
    """A non-canonical counts string that ``rleFrString`` parses to the same runs: some
    groups gain a redundant extra character (``0`` after a positive value, ``O`` after a
    negative one), with the continuation bit set on the character before it."""
    out = bytearray()
    for c in counts:
        v = c - 48
        out.append(c)
        ends_group = not v & 0x20
        if ends_group and rng.below(3) == 0:
            out[-1] = (v | 0x20) + 48
            out.append(48 + (0x1F if v & 0x10 else 0))
    return bytes(out)


def dump_rle() -> tuple[dict, dict]:
    """``rle.json`` and ``rle_iou.json`` (the IoU matrices, apart to keep each file small)."""
    families = []
    family_ious = []
    all_masks: list[list[tuple[bytearray, RleMask]]] = []
    for f in range(FAMILIES):
        rng = SplitMix64(family_seed(f))
        h, w = mask_size(rng)
        masks = [random_mask(rng, h, w) for _ in range(FAMILY_SIZE)]
        split = 1 + rng.below(FAMILY_SIZE - 1)
        crowd = [rng.below(2) for _ in range(FAMILY_SIZE - split)]
        rles = [to_rle(m, h, w) for m in masks]
        all_masks.append(list(zip(masks, rles, strict=True)))
        per_mask = []
        for m, rle in zip(masks, rles, strict=True):
            per_mask.append(
                {
                    "fnv": fnv1a64_bytes(m),
                    "counts": text(rle),
                    "area": int(mask_utils.area(rle)),
                    "bbox": bits(mask_utils.toBbox(rle)),
                    "merge_one": text(merge([rle])),
                }
            )
        families.append(
            {
                "seed": family_seed(f),
                "h": h,
                "w": w,
                "masks": per_mask,
                "split": split,
                "crowd": crowd,
                "merge_union": rle_json(merge(rles, intersect=False)),
                "merge_intersect": rle_json(merge(rles, intersect=True)),
                "pair_union": rle_json(merge(rles[:2], intersect=False)),
                "pair_intersect": rle_json(merge(rles[:2], intersect=True)),
                "overlap": bool(masks_overlap(rles)),
                "pair_overlap": bool(masks_overlap(rles[:2])),
            }
        )
        family_ious.append(
            {
                "iou_plain": iou_bits(rles, rles, [0] * FAMILY_SIZE),
                "iou_crowd": iou_bits(rles, rles, [1] * FAMILY_SIZE),
                "iou_split": iou_bits(rles[:split], rles[split:], crowd),
            }
        )
    cross = []
    cross_ious = []
    for f in range(FAMILIES - 1):
        a = [rle for _, rle in all_masks[f]]
        b = [rle for _, rle in all_masks[f + 1]]
        cross_ious.append(
            {
                "iou": iou_bits(a, b, [0] * FAMILY_SIZE),
                "ioa": iou_bits(a, b, [1] * FAMILY_SIZE),
            }
        )
        cross.append(
            {
                "merge": rle_json(merge([a[0], b[0]], intersect=False)),
                "overlap": bool(masks_overlap([a[0], b[0]])),
            }
        )
    non_canonical = []
    rng = SplitMix64(7_999_999)
    for f in range(0, FAMILIES, 5):
        for m, rle in all_masks[f][:2]:
            padded = pad_groups(text(rle).encode(), rng)
            obj: RleMask = {"size": rle["size"], "counts": padded}
            h, w = rle["size"]
            assert np.array_equal(mask_utils.decode(obj), mask_utils.decode(rle))
            non_canonical.append(
                {
                    "h": int(h),
                    "w": int(w),
                    "counts": padded.decode(),
                    "fnv": fnv1a64_bytes(m),
                    "area": int(mask_utils.area(obj)),
                    "bbox": bits(mask_utils.toBbox(obj)),
                    "merge_one": text(merge([obj])),
                }
            )
    overrun: RleMask = {"size": [2, 2], "counts": b"5"}
    try:
        mask_utils.decode(overrun)
        overrun_error = None
    except ValueError as err:
        overrun_error = str(err)
    rle = {
        "families": families,
        "cross": cross,
        "non_canonical": non_canonical,
        "overrun": {"h": 2, "w": 2, "counts": "5", "error": overrun_error},
    }
    return rle, {"families": family_ious, "cross": cross_ious}


def points_json(points: np.ndarray) -> dict:
    flat = [int(v) for v in points.ravel()]
    return {"n": len(points), "fnv": fnv1a64_ints(flat)}


def dump_boundary() -> dict:
    cases = []
    for i in range(BOUNDARY_PAIRS):
        a, ha, wa, b, hb, wb, r = boundary_case(i)
        pts_a = _boundary_points(to_rle(a, ha, wa))
        pts_b = _boundary_points(to_rle(b, hb, wb))
        match_a = int(np.count_nonzero(KDTree(pts_b).query(pts_a, k=1)[0] <= r))
        match_b = int(np.count_nonzero(KDTree(pts_a).query(pts_b, k=1)[0] <= r))
        cases.append(
            {
                "index": i,
                "fnv_a": fnv1a64_bytes(a),
                "fnv_b": fnv1a64_bytes(b),
                "r": r,
                "points_a": points_json(pts_a),
                "points_b": points_json(pts_b),
                "match_a": match_a,
                "match_b": match_b,
            }
        )
    return {"cases": cases}


def object_array(items: list) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def sequence_data(spec: dict, name: str) -> SequenceData:
    def frames(side: str):
        ids, masks = [], []
        for frame in spec[side]:
            ids.append(np.array([track for track, _, _, _ in frame], dtype=np.int64))
            masks.append(object_array([to_rle(m, mh, mw) for _, mh, mw, m in frame]))
        return tuple(ids), tuple(masks)

    gt_ids, gt_masks = frames("gt")
    pred_ids, pred_masks = frames("pred")
    return SequenceData(
        name=name,
        num_timesteps=spec["num_timesteps"],
        num_gt_ids=spec["num_gt_ids"],
        num_pred_ids=spec["num_pred_ids"],
        num_gt_dets=sum(len(f) for f in gt_ids),
        num_pred_dets=sum(len(f) for f in pred_ids),
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        pred_confidences=tuple(np.ones(len(f)) for f in pred_ids),
        geometry=MaskGeometry(gt=gt_masks, pred=pred_masks),
        similarity=tuple(np.zeros((len(g), len(p))) for g, p in zip(gt_ids, pred_ids, strict=True)),
    )


def scores_json(scores: dict) -> list:
    return [[field, bits(value)] for field, value in scores.items()]


def dump_jf() -> dict:
    metric = JAndF()
    sequences = []
    scored: list[tuple[int, dict]] = []
    for i in range(JF_SEQUENCES):
        spec = jf_sequence(i)
        flat = []
        for side in ("gt", "pred"):
            for frame in spec[side]:
                for track, mh, mw, m in frame:
                    flat.extend([track, mh, mw, fnv1a64_bytes(m)])
        record: dict[str, object] = {"index": i, "inputs_fnv": fnv1a64_ints(flat)}
        data = sequence_data(spec, f"seq{i}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scores = metric.eval_sequence(data)
        except (IndexError, ValueError, ZeroDivisionError) as err:
            record["error"] = str(err)
            record["error_type"] = type(err).__name__
        else:
            record["scores"] = scores_json(scores)
            scored.append((i, scores))
        sequences.append(record)
    groups = []
    at = 0
    k = 0
    while at < len(scored):
        size = GROUP_SIZES[k % len(GROUP_SIZES)]
        group = scored[at : at + size]
        at += size
        k += 1
        all_res = {f"seq{i}": scores for i, scores in group}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            groups.append(
                {
                    "members": [i for i, _ in group],
                    "combine_sequences": scores_json(metric.combine_sequences(all_res)),
                    "class_averaged": scores_json(metric.combine_classes_class_averaged(all_res)),
                    "det_averaged": scores_json(metric.combine_classes_det_averaged(all_res)),
                }
            )
    return {"sequences": sequences, "groups": groups}


def write(name: str, payload: dict) -> None:
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    rle, rle_iou = dump_rle()
    write("rle.json", rle)
    write("rle_iou.json", rle_iou)
    write("jf_boundary.json", dump_boundary())
    write("jf_sequences.json", dump_jf())


if __name__ == "__main__":
    main()
