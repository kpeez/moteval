import math
import struct
from collections.abc import Iterable

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
EPS = 2.0**-52


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n

    def uniform(self) -> float:
        return (self.next() >> 11) * 2.0**-53

    def spread(self) -> float:
        negative = self.next() & 1
        mantissa = self.uniform()
        exponent = self.below(86) - 30
        value = math.ldexp(mantissa, exponent)
        return -value if negative else value


def f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def fnv1a64(values: Iterable[float]) -> int:
    h = FNV_OFFSET
    for x in values:
        for byte in struct.pack("<d", x):
            h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def pattern(name: str, n: int, seed: int) -> list[float]:
    rng = SplitMix64(seed)
    if name == "alternating":
        cycle = (1e16, 1.0, -1e16, 1.0)
        return [cycle[i % 4] for i in range(n)]
    if name == "negzero":
        return [-0.0] * n
    out = []
    for _ in range(n):
        if name == "spread":
            out.append(rng.spread())
        elif name == "uniform":
            out.append(rng.uniform())
        elif name == "big_small":
            u = rng.next()
            value = 1e16 if u & 1 else 1.0
            out.append(-value if u & 2 else value)
        elif name == "nan_spread":
            value = rng.spread()
            out.append(math.nan if rng.below(5) == 0 else value)
        else:
            raise ValueError(f"unknown pattern {name!r}")
    return out


def random_box(rng: SplitMix64) -> list[float]:
    x = rng.below(40) * 0.5
    y = rng.below(40) * 0.5
    w = rng.below(24) * 0.5
    h = rng.below(24) * 0.5
    if rng.below(3) == 0:
        x += rng.uniform()
        w += rng.uniform()
    if rng.below(4) == 0:
        y += rng.uniform() * 3.0
        h += rng.uniform()
    if rng.below(10) == 0:
        w = -w
    if rng.below(12) == 0:
        x += 1048576.0
    if rng.below(15) == 0:
        x = -0.0
    return [x, y, w, h]


LSAP_CATEGORIES = (
    ("square", 1500),
    ("wide", 1500),
    ("tall", 1500),
    ("integer_ties", 1500),
    ("clear_style", 1500),
    ("identity_style", 1500),
    ("hota_style", 1000),
    ("maximize", 500),
    ("large", 100),
)


def lsap_seed(category: str, index: int) -> int:
    base = [name for name, _ in LSAP_CATEGORIES].index(category)
    return (base + 1) * 1_000_000 + index


def lsap_case(category: str, seed: int) -> tuple[int, int, list[float], bool]:
    rng = SplitMix64(seed)
    maximize = False
    if category == "square":
        nr = nc = rng.below(13)
        cost = [rng.uniform() for _ in range(nr * nc)]
    elif category == "wide":
        nr = rng.below(10)
        nc = nr + 1 + rng.below(6)
        cost = [rng.uniform() for _ in range(nr * nc)]
    elif category == "tall":
        nc = rng.below(10)
        nr = nc + 1 + rng.below(6)
        cost = [rng.uniform() for _ in range(nr * nc)]
    elif category == "integer_ties":
        nr = 1 + rng.below(10)
        nc = 1 + rng.below(10)
        cost = [float(rng.below(4)) for _ in range(nr * nc)]
    elif category == "clear_style":
        nr = 1 + rng.below(10)
        nc = 1 + rng.below(10)
        cost = []
        for _ in range(nr * nc):
            sim = rng.uniform()
            if rng.below(5) < 2:
                sim = sim * 0.5
            continues = rng.below(8) == 0
            score = 0.0 if sim < 0.5 - EPS else (1000.0 if continues else 0.0) + sim
            cost.append(-score)
    elif category == "identity_style":
        cost, nr = _identity_cost(rng)
        nc = nr
    elif category == "hota_style":
        nr = 1 + rng.below(10)
        nc = 1 + rng.below(10)
        cost = []
        for _ in range(nr * nc):
            alignment = rng.uniform()
            sim = 0.0 if rng.below(4) == 0 else rng.uniform()
            cost.append(-(alignment * sim))
    elif category == "maximize":
        nr = rng.below(9)
        nc = rng.below(9)
        cost = [rng.uniform() * 100.0 for _ in range(nr * nc)]
        maximize = True
    elif category == "large":
        nr = 20 + rng.below(41)
        nc = 20 + rng.below(41)
        ties = rng.below(2) == 0
        cost = [float(rng.below(6)) if ties else rng.uniform() for _ in range(nr * nc)]
    else:
        raise ValueError(f"unknown category {category!r}")
    return nr, nc, cost, maximize


def _identity_cost(rng: SplitMix64) -> tuple[list[float], int]:
    num_gt = rng.below(7)
    num_pred = rng.below(7)
    gt_count = [float(1 + rng.below(10)) for _ in range(num_gt)]
    pred_count = [float(1 + rng.below(10)) for _ in range(num_pred)]
    matches = []
    for g in range(num_gt):
        row = []
        for p in range(num_pred):
            if rng.below(2):
                row.append(float(rng.below(int(min(gt_count[g], pred_count[p])) + 1)))
            else:
                row.append(0.0)
        matches.append(row)
    n = num_gt + num_pred
    fp = [[0.0] * n for _ in range(n)]
    fn = [[0.0] * n for _ in range(n)]
    for r in range(num_gt, n):
        for c in range(num_pred):
            fp[r][c] = 1e10
    for r in range(num_gt):
        for c in range(num_pred, n):
            fn[r][c] = 1e10
    for g in range(num_gt):
        for c in range(num_pred):
            fn[g][c] = gt_count[g]
        fn[g][num_pred + g] = gt_count[g]
    for p in range(num_pred):
        for r in range(num_gt):
            fp[r][p] = pred_count[p]
        fp[p + num_gt][p] = pred_count[p]
    for g in range(num_gt):
        for p in range(num_pred):
            fn[g][p] -= matches[g][p]
            fp[g][p] -= matches[g][p]
    return [fn[r][c] + fp[r][c] for r in range(n) for c in range(n)], n
