"""Seeded prediction perturbation for the real-data parity gate (#20).

Generates tracker predictions from GT with drops, false positives, ID switches
and (for boxes) geometric jitter, deterministically from a fixed seed. Two
independence guarantees the gate relies on:

- Track ids are numbered independently of GT (dense remap into 1000+, switches
  and false positives allocate fresh ids from 5000+), so densification bugs
  that only work when pred ids mirror GT ids cannot hide.
- Frame numbers are derived from the declared `FrameConvention` by iterating
  timestep indices (``first_frame + t``), never copied from GT rows.

Mask perturbation preserves the MOTS invariant that predictions within a frame
are disjoint: translated masks are checked against the union of already-placed
masks and fall back to their original position (or are dropped) on conflict.
"""

from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils

from moteval.data.model import FrameConvention
from moteval.formats.mot_txt import Track
from moteval.formats.mots_txt import MaskTrack

_BASE_ID = 1000
_FRESH_ID = 5000


def perturb_box_tracks(
    gt_tracks: tuple[Track, ...],
    num_timesteps: int,
    convention: FrameConvention,
    *,
    seed: int,
    drop_rate: float = 0.15,
    fp_rate: float = 0.1,
    switch_rate: float = 0.3,
) -> list[Track]:
    """Perturbed box predictions for one sequence, reproducible from ``seed``."""
    rng = np.random.default_rng(seed)
    by_timestep: dict[int, list[Track]] = defaultdict(list)
    for t in gt_tracks:
        by_timestep[convention.to_index(t.frame, num_timesteps)].append(t)

    id_map = {
        raw: _BASE_ID + dense for dense, raw in enumerate(sorted({t.track_id for t in gt_tracks}))
    }
    switched: dict[int, int] = {}
    switch_at: dict[int, int] = {
        raw: int(rng.integers(1, num_timesteps))
        for raw in sorted(id_map)
        if num_timesteps > 1 and rng.random() < switch_rate
    }
    next_fresh = _FRESH_ID

    preds: list[Track] = []
    for t_index in range(num_timesteps):
        frame = convention.first_frame + t_index
        for gt_row in by_timestep.get(t_index, []):
            if rng.random() < drop_rate:
                continue
            raw = gt_row.track_id
            if raw in switch_at and t_index >= switch_at[raw] and raw not in switched:
                switched[raw] = next_fresh
                next_fresh += 1
            pred_id = switched.get(raw, id_map[raw])
            dx, dy = rng.normal(0.0, 2.0, size=2)
            scale_w, scale_h = 1.0 + rng.normal(0.0, 0.05, size=2)
            preds.append(
                Track(
                    frame=frame,
                    track_id=pred_id,
                    x=gt_row.x + dx,
                    y=gt_row.y + dy,
                    w=max(1.0, gt_row.w * scale_w),
                    h=max(1.0, gt_row.h * scale_h),
                    conf=float(rng.uniform(0.5, 1.0)),
                )
            )
        if rng.random() < fp_rate:
            preds.append(
                Track(
                    frame=frame,
                    track_id=next_fresh,
                    x=float(rng.uniform(0, 800)),
                    y=float(rng.uniform(0, 500)),
                    w=float(rng.uniform(20, 80)),
                    h=float(rng.uniform(40, 120)),
                    conf=float(rng.uniform(0.5, 1.0)),
                )
            )
            next_fresh += 1
    return preds


def _translate_rle(rle_dict: dict, dy: int, dx: int) -> dict:
    mask = mask_utils.decode(rle_dict)
    mask = np.roll(mask, (dy, dx), axis=(0, 1))
    return mask_utils.encode(np.asfortranarray(mask))


def _overlaps(rle_dict: dict, occupied: list[dict]) -> bool:
    if not occupied:
        return False
    merged = mask_utils.merge(occupied, intersect=False)
    inter = mask_utils.merge([rle_dict, merged], intersect=True)
    return float(mask_utils.area(inter)) > 0.0


def perturb_mask_tracks(
    gt_tracks: tuple[MaskTrack, ...],
    num_timesteps: int,
    convention: FrameConvention,
    *,
    seed: int,
    drop_rate: float = 0.15,
    switch_rate: float = 0.3,
    max_shift: int = 4,
) -> list[MaskTrack]:
    """Perturbed MOTS predictions: drops, switches, small translations.

    Frame-local disjointness is preserved: a translated mask that would overlap
    an already-placed prediction keeps its original position instead, and is
    dropped if even that conflicts (cannot happen for disjoint GT, but guards
    the invariant).
    """
    rng = np.random.default_rng(seed)
    by_timestep: dict[int, list[MaskTrack]] = defaultdict(list)
    for t in gt_tracks:
        by_timestep[convention.to_index(t.frame, num_timesteps)].append(t)

    id_map = {
        raw: _BASE_ID + dense for dense, raw in enumerate(sorted({t.track_id for t in gt_tracks}))
    }
    switched: dict[int, int] = {}
    switch_at: dict[int, int] = {
        raw: int(rng.integers(1, num_timesteps))
        for raw in sorted(id_map)
        if num_timesteps > 1 and rng.random() < switch_rate
    }
    next_fresh = _FRESH_ID

    preds: list[MaskTrack] = []
    for t_index in range(num_timesteps):
        frame = convention.first_frame + t_index
        occupied: list[dict] = []
        for gt_row in by_timestep.get(t_index, []):
            if rng.random() < drop_rate:
                continue
            raw = gt_row.track_id
            if raw in switch_at and t_index >= switch_at[raw] and raw not in switched:
                switched[raw] = next_fresh
                next_fresh += 1
            pred_id = switched.get(raw, id_map[raw])

            original = {"size": [gt_row.img_h, gt_row.img_w], "counts": gt_row.rle.encode()}
            dy, dx = (int(v) for v in rng.integers(-max_shift, max_shift + 1, size=2))
            candidate = _translate_rle(original, dy, dx)
            if _overlaps(candidate, occupied):
                candidate = original
            if _overlaps(candidate, occupied):
                continue
            occupied.append(candidate)
            counts = candidate["counts"]
            assert isinstance(counts, bytes)
            preds.append(
                MaskTrack(
                    frame=frame,
                    track_id=pred_id,
                    class_id=gt_row.class_id,
                    img_h=gt_row.img_h,
                    img_w=gt_row.img_w,
                    rle=counts.decode(),
                )
            )
    return preds
