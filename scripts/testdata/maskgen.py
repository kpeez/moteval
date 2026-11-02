from scripts.testdata.testgen import SplitMix64

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def fnv1a64_bytes(data: bytes | bytearray) -> int:
    h = FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def fnv1a64_ints(values: list[int]) -> int:
    h = FNV_OFFSET
    for value in values:
        for byte in (value & MASK64).to_bytes(8, "little"):
            h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def mask_size(rng: SplitMix64) -> tuple[int, int]:
    if rng.below(8) == 0:
        return 40 + rng.below(160), 40 + rng.below(160)
    return 1 + rng.below(24), 1 + rng.below(24)


# Masks are column-major (pixel (r, c) at c * h + r), the order pycocotools encodes.
def _fill(m: bytearray, h: int, r0: int, r1: int, c0: int, c1: int, value: int) -> None:
    run = bytes([value]) * (r1 - r0)
    for c in range(c0, c1):
        m[c * h + r0 : c * h + r1] = run


def _rect(rng: SplitMix64, h: int, w: int) -> tuple[int, int, int, int]:
    r0 = rng.below(h)
    c0 = rng.below(w)
    r1 = r0 + 1 + rng.below(h - r0)
    c1 = c0 + 1 + rng.below(w - c0)
    return r0, r1, c0, c1


def random_mask(rng: SplitMix64, h: int, w: int, allow_noise: bool = True) -> bytearray:
    m = bytearray(h * w)
    kind = rng.below(8)
    if kind == 4 and not allow_noise:
        kind = 3
    if kind == 0:
        return m
    if kind == 1:
        return bytearray(b"\x01" * (h * w))
    if kind == 2:
        r = rng.below(h)
        c = rng.below(w)
        m[c * h + r] = 1
        return m
    if kind in (3, 6):
        value = 1
        if kind == 6:
            m = bytearray(b"\x01" * (h * w))
            value = 0
        for _ in range(1 + rng.below(4)):
            r0, r1, c0, c1 = _rect(rng, h, w)
            _fill(m, h, r0, r1, c0, c1, value)
        return m
    if kind == 4:
        p = 1 + rng.below(7)
        for i in range(h * w):
            if rng.below(8) < p:
                m[i] = 1
        return m
    if kind == 5:
        for c in range(w):
            if rng.below(3) != 0:
                a = rng.below(h)
                b = a + rng.below(h - a + 1)
                _fill(m, h, a, b, c, c + 1, 1)
        return m
    for _ in range(1 + rng.below(3)):
        r0, r1, c0, c1 = _rect(rng, h, w)
        side = rng.below(4)
        if side == 0:
            r0 = 0
        elif side == 1:
            r1 = h
        elif side == 2:
            c0 = 0
        else:
            c1 = w
        _fill(m, h, r0, r1, c0, c1, 1)
    return m


def mask_area(m: bytearray) -> int:
    return sum(m)


FAMILIES = 250
FAMILY_SIZE = 4


def family_seed(index: int) -> int:
    return 7_000_000 + index


def boundary_seed(index: int) -> int:
    return 8_000_000 + index


def jf_sequence_seed(index: int) -> int:
    return 9_000_000 + index


def nonempty_mask(rng: SplitMix64, h: int, w: int) -> bytearray:
    while True:
        m = random_mask(rng, h, w)
        if 0 < mask_area(m) < h * w:
            return m


def boundary_case(index: int) -> tuple[bytearray, int, int, bytearray, int, int, int]:
    rng = SplitMix64(boundary_seed(index))
    if rng.below(5) == 0:
        ha, wa = 60 + rng.below(200), 60 + rng.below(200)
    else:
        ha, wa = 2 + rng.below(60), 2 + rng.below(60)
    hb, wb = ha, wa
    if rng.below(5) == 0:
        hb, wb = 2 + rng.below(60), 2 + rng.below(60)
    a = nonempty_mask(rng, ha, wa)
    b = nonempty_mask(rng, hb, wb)
    r = 1 + rng.below(12)
    return a, ha, wa, b, hb, wb, r


def jf_sequence(index: int) -> dict:
    rng = SplitMix64(jf_sequence_seed(index))
    big = rng.below(12) == 0
    if big:
        h, w = 400 + rng.below(800), 400 + rng.below(1200)
        num_timesteps = 1 + rng.below(3)
    else:
        h, w = 2 + rng.below(20), 2 + rng.below(20)
        num_timesteps = 250 + rng.below(12) if rng.below(6) == 0 else 1 + rng.below(25)
    num_gt_ids = rng.below(5)
    num_pred_ids = rng.below(6)
    density = 1 + rng.below(4)
    gt: list[list[tuple[int, int, int, bytearray]]] = []
    pred: list[list[tuple[int, int, int, bytearray]]] = []
    for _ in range(num_timesteps):
        for side, count in ((gt, num_gt_ids), (pred, num_pred_ids)):
            frame = []
            for track in range(count):
                if rng.below(4) < density:
                    mh, mw = (h + 1, w) if rng.below(40) == 0 else (h, w)
                    frame.append((track, mh, mw, random_mask(rng, mh, mw, allow_noise=not big)))
            side.append(frame)
    return {
        "h": h,
        "w": w,
        "num_timesteps": num_timesteps,
        "num_gt_ids": num_gt_ids,
        "num_pred_ids": num_pred_ids,
        "gt": gt,
        "pred": pred,
    }
