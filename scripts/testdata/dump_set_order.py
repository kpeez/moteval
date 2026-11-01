import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.testdata.reference import require_reference

require_reference()

from moteval.metrics.track_map import _union_frames  # pyrefly: ignore[missing-module-attribute]
from scripts.testdata.testgen import MASK64, SplitMix64

OUT = Path(__file__).resolve().parents[2] / "tests" / "integration" / "data" / "set_order.json"

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
SEPARATOR = MASK64

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
    h = FNV_OFFSET
    for key in keys:
        for byte in key.to_bytes(8, "little"):
            h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def distinct(rng: SplitMix64, n: int, span: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    while len(out) < n:
        key = rng.below(span)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def shuffle(rng: SplitMix64, keys: list[int]) -> None:
    for i in range(len(keys) - 1, 0, -1):
        j = rng.below(i + 1)
        keys[i], keys[j] = keys[j], keys[i]


def set_order_case(regime: str, case_seed: int) -> tuple[list[int], list[int]]:
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
