"""Dump CPython's iteration order of TrackMAP's frame-union sets for the Rust port.

TrackMAP's track IoU sums a pair's frames in the order ``_union_frames`` iterates
(``set(gt.keys()) | set(dt.keys())``), so the Rust port replays CPython's set hash table
(`moteval::metrics::track_map::union_frame_order`). This script draws seeded key sets
and insertion orders, runs them through moteval's own ``_union_frames``, and writes
``rust/testdata/set_order.json``. ``rust/moteval/tests/set_order.rs`` regenerates each
case from its seed with the same generator and requires the same order. Run with the
reference interpreter (CPython 3.14.8):

    uv run python rust/scripts/dump_set_order.py

Cases are grouped by regime; case ``i`` of a regime uses seed ``first_seed + i``. Each
case is one row: ``inputs_fnv`` (FNV-1a 64 over the gt keys, a separator, and the dt keys,
each as little-endian ``u64``), the order's ``len`` and ``order_fnv``, and ``ascending``
(1 when the set iterates in ascending order, the case Python's vectorized path covers).
"""

import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moteval.metrics.track_map import _union_frames
from rust.scripts.testgen import MASK64, SplitMix64

OUT = Path(__file__).resolve().parents[1] / "testdata" / "set_order.json"

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
SEPARATOR = MASK64

# (regime, number of cases). Case ``i`` of a regime draws from ``SplitMix64(seed(r, i))``.
REGIMES = (
    ("dense_ascending", 600),
    ("sparse_ascending", 600),
    ("sparse_shuffled", 600),
    ("colliding", 600),
    ("one_side_empty", 300),
    ("subset_overlap", 600),
    ("short_far_tracks", 1000),
    ("huge", 4),
)
SPANS = (16, 64, 256, 4096, 100_000)
_BOX = np.zeros(4)


def seed(regime: str, index: int) -> int:
    base = [name for name, _ in REGIMES].index(regime)
    return 9_000_000 + base * 100_000 + index


def fnv1a64_keys(keys: list[int]) -> int:
    """FNV-1a over each key's little-endian ``u64`` bytes."""
    h = FNV_OFFSET
    for key in keys:
        for byte in key.to_bytes(8, "little"):
            h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def distinct(rng: SplitMix64, n: int, span: int) -> list[int]:
    """``n`` distinct keys below ``span``, in draw order."""
    seen: set[int] = set()
    out: list[int] = []
    while len(out) < n:
        key = rng.below(span)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def shuffle(rng: SplitMix64, keys: list[int]) -> None:
    """Fisher-Yates, from the last position down."""
    for i in range(len(keys) - 1, 0, -1):
        j = rng.below(i + 1)
        keys[i], keys[j] = keys[j], keys[i]


def set_order_case(regime: str, case_seed: int) -> tuple[list[int], list[int]]:
    """``(gt_keys, dt_keys)``, each in dict insertion order."""
    rng = SplitMix64(case_seed)
    if regime == "dense_ascending":
        g_start = rng.below(3000)
        g_len = rng.below(200)
        d_start = max(0, g_start + rng.below(g_len + 41) - 20)
        d_len = rng.below(200)
        return list(range(g_start, g_start + g_len)), list(range(d_start, d_start + d_len))
    if regime in ("sparse_ascending", "sparse_shuffled"):
        span = SPANS[rng.below(len(SPANS))]
        cap = min(span // 2, 300)
        gt = distinct(rng, rng.below(cap + 1), span)
        dt = distinct(rng, rng.below(cap + 1), span)
        if regime == "sparse_ascending":
            gt.sort()
            dt.sort()
        return gt, dt
    if regime == "colliding":
        base = 1 << (3 + rng.below(10))
        sides = []
        for _ in range(2):
            n = rng.below(40)
            seen: set[int] = set()
            keys: list[int] = []
            while len(keys) < n:
                key = rng.below(3) + base * rng.below(64)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
            if rng.below(2):
                keys.sort()
            sides.append(keys)
        return sides[0], sides[1]
    if regime == "one_side_empty":
        span = SPANS[rng.below(len(SPANS))]
        cap = min(span // 2, 300)
        keys = distinct(rng, 1 + rng.below(cap), span)
        which = rng.below(3)
        if which == 0:
            return keys, []
        if which == 1:
            return [], keys
        return [], []
    if regime == "subset_overlap":
        span = SPANS[rng.below(len(SPANS))]
        cap = min(span // 2, 300)
        gt = distinct(rng, rng.below(cap + 1), span)
        dt = [key for key in gt if rng.below(2) == 0]
        shuffle(rng, dt)
        dt_seen = set(dt)
        for _ in range(rng.below(10)):
            key = rng.below(span)
            if key not in dt_seen:
                dt_seen.add(key)
                dt.append(key)
        return gt, dt
    if regime == "short_far_tracks":
        num_frames = 100 + rng.below(5000)
        g_start = rng.below(num_frames)
        g_len = 1 + rng.below(8)
        d_start = max(0, g_start + rng.below(17) - 8)
        d_len = 1 + rng.below(8)
        gt = list(range(g_start, min(g_start + g_len, num_frames)))
        return gt, list(range(d_start, d_start + d_len))
    if regime == "huge":
        # Above 50,000 used slots CPython grows a set 2x instead of 4x.
        gt = distinct(rng, 50_000 + rng.below(20_000), 1_000_000)
        if rng.below(2):
            gt.sort()
        dt = distinct(rng, 1 + rng.below(5000), 1_000_000)
        return gt, dt
    raise ValueError(f"unknown regime {regime!r}")


def main() -> None:
    regimes = []
    total = 0
    non_ascending = 0
    for regime, count in REGIMES:
        rows = []
        for index in range(count):
            gt_keys, dt_keys = set_order_case(regime, seed(regime, index))
            gt_frames = {key: _BOX for key in gt_keys}
            dt_frames = {key: _BOX for key in dt_keys}
            order = list(_union_frames(dt_frames, gt_frames))
            ascending = all(b > a for a, b in itertools.pairwise(order))
            non_ascending += not ascending
            inputs_fnv = fnv1a64_keys([*gt_keys, SEPARATOR, *dt_keys])
            rows.append([inputs_fnv, len(order), fnv1a64_keys(order), int(ascending)])
        regimes.append({"regime": regime, "first_seed": seed(regime, 0), "cases": rows})
        total += count
    body = {
        "format": 1,
        "python": platform.python_version(),
        "row": ["inputs_fnv", "len", "order_fnv", "ascending"],
        "non_ascending": non_ascending,
        "regimes": regimes,
    }
    OUT.write_text(json.dumps(body, separators=(",", ":")) + "\n")
    print(f"wrote {total} cases ({non_ascending} not ascending) to {OUT}")


if __name__ == "__main__":
    main()
