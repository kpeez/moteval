"""Dump numpy's reductions, ranges, rounding, log10, and moteval's box similarity as bits.

Writes ``rust/testdata/numeric.json`` and ``rust/testdata/box_similarity.json``, which
``rust/moteval/tests/numeric.rs`` and ``tests/box_similarity.rs`` check bit for bit.
Inputs come from ``testgen.py`` (seeded, regenerated in Rust); floats are stored as
``u64`` bit patterns. Run with the reference interpreter and numpy 2.5.1:

    artifacts/rust-port/ref-venv/bin/python rust/scripts/dump_numeric.py
"""

import functools
import json
import operator
import struct
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
# After site-packages: `moteval` is the pure-Python reference (rust/scripts/reference.py).
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from rust.scripts.reference import require_reference

require_reference()

from moteval.data.similarity import box_ioa, box_iou  # pyrefly: ignore[missing-import]

from rust.scripts.testgen import SplitMix64, f64_bits, fnv1a64, pattern, random_box

OUT_DIR = Path(__file__).resolve().parents[1] / "testdata"

LENGTHS = (
    *range(18),
    24, 25, 31, 32, 33, 63, 64, 65, 127, 128, 129, 130, 136, 137, 255, 256, 257,
    1000, 8191, 8192, 8193, 100003,
)  # fmt: skip
PATTERNS = ("alternating", "spread", "big_small", "negzero", "uniform")
SHAPES = ((1, 1), (1, 19), (3, 8), (5, 9), (25, 19), (45, 10), (7, 129), (130, 3))
LARGE_SHAPE = (300, 301)  # spread pattern only, to keep the file small
# One column: numpy reduces axis 0 of an (n, 1) array pairwise, not row by row. Appended
# after the other shapes so their seeds do not change.
COLUMN_SHAPES = ((8, 1), (9, 1), (25, 1), (130, 1))


def bits(values) -> list[int]:
    return [f64_bits(float(x)) for x in np.asarray(values, dtype=np.float64).ravel()]


def reductions() -> list[dict]:
    cases = []
    seed = 1000
    for name in PATTERNS:
        for n in LENGTHS:
            seed += 1
            values = pattern(name, n, seed)
            a = np.array(values, dtype=np.float64)
            case = {
                "pattern": name,
                "n": n,
                "seed": seed,
                "inputs_fnv": fnv1a64(values),
                "sum": f64_bits(float(np.sum(a))),
                "mean": f64_bits(float(np.mean(a))) if n else None,
                "cumsum": bits(np.cumsum(a)) if n <= 129 else None,
            }
            cases.append(case)
    return cases


def nanmeans() -> list[dict]:
    cases = []
    seed = 2000
    for n in (0, 1, 2, 7, 8, 9, 25, 64, 128, 129, 1000):
        for name in ("nan_spread", "spread"):
            seed += 1
            values = pattern(name, n, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # empty and all-NaN slices
                value = float(np.nanmean(np.array(values, dtype=np.float64)))
            cases.append(
                {
                    "pattern": name,
                    "n": n,
                    "seed": seed,
                    "inputs_fnv": fnv1a64(values),
                    "value": f64_bits(value),
                }
            )
    return cases


def axes() -> list[dict]:
    cases = []
    seed = 3000
    work = [(name, shape) for name in ("spread", "big_small", "negzero") for shape in SHAPES]
    columns = [(name, shape) for name in ("spread", "big_small") for shape in COLUMN_SHAPES]
    for name, (rows, cols) in [*work, ("spread", LARGE_SHAPE), *columns]:
        seed += 1
        values = pattern(name, rows * cols, seed)
        m = np.array(values, dtype=np.float64).reshape(rows, cols)
        cases.append(
            {
                "pattern": name,
                "rows": rows,
                "cols": cols,
                "seed": seed,
                "inputs_fnv": fnv1a64(values),
                "sum_all": f64_bits(float(np.sum(m))),
                "sum_axis0": bits(np.sum(m, axis=0)),
                "mean_axis0": bits(np.mean(m, axis=0)),
                "sum_last_axis": bits(np.sum(m, axis=-1)),
                "mean_last_axis": bits(np.mean(m, axis=-1)),
            }
        )
    return cases


def averages() -> list[dict]:
    cases = []
    seed = 4000
    for n in (1, 2, 3, 9, 25, 130, 1000):
        seed += 1
        a = pattern("spread", n, seed)
        w = pattern("uniform", n, seed + 500)
        cases.append(
            {
                "n": n,
                "seed": seed,
                "weights_seed": seed + 500,
                "inputs_fnv": fnv1a64(a + w),
                "value": f64_bits(float(np.average(np.array(a), weights=np.array(w)))),
            }
        )
    return cases


def aranges() -> list[dict]:
    args = (
        (0.05, 0.99, 0.05),  # HOTA alphas
        (0.5, 0.96, 0.05),  # TrackMAP IoU thresholds
        (0.0, 1.0, 0.1),
        (0.0, 1.0, 0.01),
        (1.0, 2.0, 0.3),
        (-1.0, 1.0, 0.25),
        (1.0, 0.0, -0.1),
        (0.0, 1.0, 2.0),
        (1.0, 1.0, 0.5),
        (0.1, 0.30000000000000004, 0.1),
    )
    return [
        {
            "start": f64_bits(s),
            "stop": f64_bits(e),
            "step": f64_bits(d),
            "out": bits(np.arange(s, e, d)),
        }
        for s, e, d in args
    ]


def linspaces() -> list[dict]:
    args: list[tuple[float, float, int, bool]] = [
        (0.0, 1.0, int(np.round((1.00 - 0.0) / 0.01) + 1), True),  # TrackMAP recall
        (0.0, 1.0, 10, False),
        (0.0, 0.0, 5, True),
        (5.0, 5.0, 1, True),
        (0.0, 1.0, 1, False),
        (0.0, 1.0, 0, True),
        (2.0, 1e-320, 4, True),
        (0.0, 5e-324, 3, True),
    ]
    args += [(1.0, float(t), 5, True) for t in range(1, 601)]  # J&F decay bins
    return [
        {
            "start": f64_bits(s),
            "stop": f64_bits(e),
            "num": num,
            "endpoint": endpoint,
            "out": bits(np.linspace(s, e, num, endpoint=endpoint)),
        }
        for s, e, num, endpoint in args
    ]


def roundings() -> list[dict]:
    xs = [
        0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999999999999994, 0.5000000000000001,
        4503599627370495.5, 4503599627370497.0, -0.0, 0.0, 1e300, -1e-300,
        float("inf"), float("-inf"), float("nan"), 100.0000000001, 3.4999999999999996,
    ]  # fmt: skip
    for t in range(1, 601):
        xs.extend((np.linspace(1, t, 5) + 1e-10).tolist())
    return [{"x": f64_bits(x), "out": f64_bits(float(np.round(x)))} for x in xs]


def extrema() -> list[dict]:
    specials = [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), float("-inf"), 5e-324]
    return [
        {
            "a": f64_bits(a),
            "b": f64_bits(b),
            "maximum": f64_bits(float(np.maximum(a, b))),
            "minimum": f64_bits(float(np.minimum(a, b))),
            "py_max": f64_bits(float(max(a, b))),
            "py_min": f64_bits(float(min(a, b))),
        }
        for a in specials
        for b in specials
    ]


def plain_sums() -> list[dict]:
    cases = []
    seed = 5000
    for name in ("spread", "big_small", "negzero", "alternating"):
        for n in (0, 1, 2, 3, 9, 25, 129):
            seed += 1
            values = pattern(name, n, seed)
            total = functools.reduce(operator.add, values, 0)
            cases.append(
                {
                    "pattern": name,
                    "n": n,
                    "seed": seed,
                    "inputs_fnv": fnv1a64(values),
                    "value": f64_bits(float(total)),
                }
            )
    return cases


LOG10_INTEGERS = 1_000_000  # the fixture keeps a fingerprint of np.log10(1..N)
RCP14_SENSITIVE = (
    0x3FF040F0DAE11493, 0x3FF15B38BC4B53C2, 0x40329E477EC11054, 0x3FB3522A792225F0,
    0x3FC41416CD5B8972, 0x4015C989A889F292, 0x4016C155ADAA507E, 0x3FA7D06F7954D294,
    0x40F8F9C0BF3D3683, 0x3FEA41A32CEF80DE, 0x3F1BACFE59930D22, 0x3FED41C188011B15,
    0x3FEF07E1F8AC6968,
)  # fmt: skip


def log10s() -> dict:
    """``np.log10`` on finite ``x > 0``, which CLEAR's MOTAL takes of ``IDSW``.

    numpy computes it with Intel SVML on an AVX-512 (``X86_V4``) Linux host and with libm
    elsewhere; the two differ in the last bit for some integers. ``numeric::log10_svml``
    ports the SVML kernel, so this fixture must come from the SVML path.
    """
    from numpy._core._multiarray_umath import (  # pyrefly: ignore[missing-import]
        __cpu_features__,
    )

    if not (sys.platform == "linux" and __cpu_features__.get("X86_V4")):
        sys.exit("dump the log10 fixture on a Linux host with X86_V4: numpy must use SVML")
    xs = [float(i) for i in range(1, 1025)]
    rng = SplitMix64(7000)
    for _ in range(1000):  # positive normals across every binade
        exponent = 1 + rng.below(2046)
        xs.append(struct.unpack("<d", struct.pack("<Q", (exponent << 52) | (rng.next() >> 12)))[0])
    # Mantissas within 2**-14 of 64/d for odd d in 33..63: where vrcp14pd, rounded to a
    # multiple of 1/32, switches between two values.
    for d in range(33, 64, 2):
        for j in range(-24, 25):
            m = 64 / d * (1 + j * 2.0**-17)
            xs.extend((m, m * 2.0**10, m * 2.0**-3))
    # Inputs where vrcp14pd and an exact 1/m round to different multiples of 1/32 and
    # SVML's result changes with it (found by a search on the reference host).
    xs.extend(struct.unpack("<d", struct.pack("<Q", b))[0] for b in RCP14_SENSITIVE)
    xs.extend((5e-324, 2.2250738585072009e-308, 1e-310, 2.2250738585072014e-308, 0.1, 1e-300))
    xs.extend((10.0, 1e22, 2.0**52, 2.0**1023, 1.7976931348623157e308, 1.0 + 2.0**-52))
    ints = np.arange(1, LOG10_INTEGERS + 1, dtype=np.float64)
    return {
        "kernel": "svml",
        "x": [f64_bits(x) for x in xs],
        "out": bits(np.log10(np.array(xs, dtype=np.float64))),
        "integers": LOG10_INTEGERS,
        "integers_fnv": fnv1a64(np.log10(ints).tolist()),
    }


def box_cases() -> list[dict]:
    cases = []
    for case in range(400):
        seed = 6000 + case
        rng = SplitMix64(seed)
        n = rng.below(7)
        m = rng.below(7)
        a = [random_box(rng) for _ in range(n)]
        b = [random_box(rng) for _ in range(m)]
        arr_a = np.array(a, dtype=np.float64).reshape(-1, 4)
        arr_b = np.array(b, dtype=np.float64).reshape(-1, 4)
        cases.append(
            {
                "seed": seed,
                "n": n,
                "m": m,
                "inputs_fnv": fnv1a64([v for box in a + b for v in box]),
                "iou": bits(box_iou(arr_a, arr_b)),
                "ioa": bits(box_ioa(arr_a, arr_b)),
            }
        )
    return cases


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    numeric = {
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "reductions": reductions(),
        "nanmean": nanmeans(),
        "axes": axes(),
        "average": averages(),
        "plain_sum": plain_sums(),
        "arange": aranges(),
        "linspace": linspaces(),
        "round": roundings(),
        "extrema": extrema(),
        "log10": log10s(),
    }
    (OUT_DIR / "numeric.json").write_text(json.dumps(numeric, separators=(",", ":")) + "\n")
    boxes = {"numpy": np.__version__, "cases": box_cases()}
    (OUT_DIR / "box_similarity.json").write_text(json.dumps(boxes, separators=(",", ":")) + "\n")
    for name in ("numeric.json", "box_similarity.json"):
        print(f"wrote {OUT_DIR / name} ({(OUT_DIR / name).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
