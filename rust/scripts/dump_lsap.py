"""Dump scipy's `linear_sum_assignment` on seeded cost matrices for the Rust port.

Writes one ``rust/testdata/lsap/<category>.json`` per category in
``testgen.LSAP_CATEGORIES`` (10,600 matrices: square, wide, tall, integer-valued with
many ties, CLEAR-style, Identity-style, HOTA-style, maximize, and large), plus
``special.json`` with explicit matrices for the error and infinity paths.
``rust/moteval/tests/lsap.rs`` regenerates each matrix from its seed and requires the same
row and column indices. Run with the reference interpreter and scipy 1.18.0:

    uv run python rust/scripts/dump_lsap.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rust.scripts.testgen import (
    LSAP_CATEGORIES,
    f64_bits,
    fnv1a64,
    lsap_case,
    lsap_seed,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "testdata" / "lsap"

INF = math.inf
NAN = math.nan
# (name, nr, nc, values, maximize)
SPECIAL: tuple[tuple[str, int, int, list[float], bool], ...] = (
    ("empty_rows", 0, 3, [], False),
    ("empty_cols", 3, 0, [], False),
    ("nan_entry", 2, 2, [0.0, NAN, 1.0, 2.0], False),
    ("neg_inf_entry", 2, 2, [0.0, -INF, 1.0, 2.0], False),
    ("pos_inf_maximize", 1, 2, [INF, 1.0], True),
    ("all_inf", 2, 2, [INF, INF, INF, INF], False),
    ("inf_row", 2, 2, [INF, INF, 1.0, 2.0], False),
    ("inf_feasible", 2, 2, [INF, 1.0, 2.0, INF], False),
    ("inf_tall_feasible", 3, 2, [INF, 1.0, INF, INF, 2.0, INF], False),
    ("constant", 4, 4, [7.0] * 16, False),
    ("negative_zero", 2, 3, [-0.0, 0.0, -0.0, 0.0, -0.0, -0.0], False),
    ("huge", 2, 2, [1e300, -1e300, -1e300, 1e300], False),
)


def solve(nr: int, nc: int, cost: list[float], maximize: bool) -> dict:
    matrix = np.array(cost, dtype=np.float64).reshape(nr, nc)
    try:
        rows, cols = linear_sum_assignment(matrix, maximize=maximize)
    except ValueError as err:
        return {"error": str(err)}
    return {"rows": rows.tolist(), "cols": cols.tolist()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for category, count in LSAP_CATEGORIES:
        cases = []
        for index in range(count):
            seed = lsap_seed(category, index)
            nr, nc, cost, maximize = lsap_case(category, seed)
            case = {"i": index, "nr": nr, "nc": nc, "fnv": fnv1a64(cost)}
            case.update(solve(nr, nc, cost, maximize))
            cases.append(case)
        body = {"scipy": scipy.__version__, "category": category, "cases": cases}
        path = OUT_DIR / f"{category}.json"
        path.write_text(json.dumps(body, separators=(",", ":")) + "\n")
        total += count
        print(f"wrote {path} ({count} cases, {path.stat().st_size} bytes)")
    special = []
    for name, nr, nc, cost, maximize in SPECIAL:
        case = {
            "name": name,
            "nr": nr,
            "nc": nc,
            "maximize": maximize,
            "bits": [f64_bits(x) for x in cost],
        }
        case.update(solve(nr, nc, cost, maximize))
        special.append(case)
    path = OUT_DIR / "special.json"
    path.write_text(json.dumps({"scipy": scipy.__version__, "cases": special}, indent=1) + "\n")
    print(f"wrote {total} seeded and {len(special)} special cases")


if __name__ == "__main__":
    main()
