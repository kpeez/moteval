import csv
import io
import json
import math
import struct
import sys
from pathlib import Path, PurePosixPath

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.testdata.testgen import FNV_OFFSET, FNV_PRIME, MASK64, SplitMix64, f64_bits, fnv1a64

OUT = Path(__file__).resolve().parents[2] / "tests" / "integration" / "data" / "pyfmt.json"
CHUNK = 100


def from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def fnv_text(texts: list[str]) -> int:
    h = FNV_OFFSET
    for text in texts:
        for byte in text.encode() + b"\n":
            h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def gen_bits(rng: SplitMix64) -> float:
    return from_bits(rng.next())


def gen_decimal(rng: SplitMix64) -> float:
    ndigits = 1 + rng.below(17)
    mantissa = rng.below(10**ndigits)
    exponent = rng.below(61) - 30
    value = float(f"{mantissa}e{exponent}")
    return -value if rng.next() & 1 else value


def gen_ratio(rng: SplitMix64) -> float:
    n = 1 + rng.below(10000)
    k = rng.below(n + 1)
    ratio = k / n
    return ratio if rng.below(2) == 0 else 100.0 * ratio


def gen_tie5(rng: SplitMix64) -> float:
    mantissa = (10000 + rng.below(90000)) * 10 + 5
    exponent = rng.below(41) - 20
    return float(f"{mantissa}e{exponent}")


def gen_dyadic(rng: SplitMix64) -> float:
    mantissa = rng.below(1 << 24)
    shift = rng.below(40)
    return math.ldexp(float(mantissa), -shift)


def gen_wide_dyadic(rng: SplitMix64) -> float:
    mantissa = (1 << 52) + rng.below(1 << 52)
    shift = rng.below(12)
    return math.ldexp(float(mantissa), -shift)


SETS = (
    ("bits", 9001, 40000, gen_bits),
    ("decimal", 9002, 20000, gen_decimal),
    ("ratio", 9003, 20000, gen_ratio),
    ("tie5", 9004, 10000, gen_tie5),
    ("dyadic", 9005, 10000, gen_dyadic),
    ("wide_dyadic", 9006, 10000, gen_wide_dyadic),
)


def seeded_sets() -> list[dict]:
    out = []
    for name, seed, n, gen in SETS:
        rng = SplitMix64(seed)
        values = [gen(rng) for _ in range(n)]
        reprs = [repr(x) for x in values]
        g5 = [format(x, "1.5g") for x in values]
        out.append(
            {
                "name": name,
                "seed": seed,
                "n": n,
                "inputs_fnv": fnv1a64(values),
                "repr_chunks": [fnv_text(reprs[i : i + CHUNK]) for i in range(0, n, CHUNK)],
                "g5_chunks": [fnv_text(g5[i : i + CHUNK]) for i in range(0, n, CHUNK)],
            }
        )
    return out


def edge_values() -> list[float]:
    values = [
        0.0, 1.0, 0.5, 0.1, 0.2, 0.1 + 0.2, 1 / 3, 2 / 3, 10.0, 100.0, 1e15, 1e16, 1e17,
        9999999999999998.0, 9999999999999999.0, 1e16 - 2, 123456789012345678.0,
        1e-4, 1e-5, 0.0001, 0.00001, 0.00012345, 0.000099999, 9.9999e-5, 1.5e-5,
        5e-324, 1e-323, 2.2250738585072014e-308, 2.225073858507201e-308,
        1.7976931348623157e308, 1e22, 1e23, 1e21, 2.0**53, 2.0**53 + 2, 2.0**63, 2.0**64,
        99999.5, 99999.49999, 99995.0, 99994.99999, 12345.5, 12344.5, 1.03125, 1.09375,
        0.000125, 0.00011115, 123455.0, 123465.0, 0.30000000000000004, 59.161, 75.0,
        46.666666666666664, 47.61904761904762, 81.64965809277261, 70.71067811865476,
        76.17986810571369, 58.33333333333333, 75.59289460184544, 57.14285714285714,
        1e-7, 1e-300, 1e300, 4.35, 2.675, 0.125, 0.375, 1e5, 99999.0, 100000.0, 999995.0,
        math.pi, math.e, math.inf, -math.inf, math.nan,
        from_bits(0x7FF8000000000001), from_bits(0xFFF8000000000000),
        from_bits(0x428DC635C8F01440),
    ]  # fmt: skip
    return values + [-x for x in values if not math.isnan(x)] + [-0.0]


def edge_cases() -> list[dict]:
    return [
        {
            "bits": f64_bits(x),
            "repr": repr(x),
            "str": str(x),
            "g5": format(x, "1.5g"),
            "json": json.dumps(x),
            "json_list": json.dumps([x, x]),
        }
        for x in edge_values()
    ]


STRINGS = [
    "", "toy-0001", "COMBINED", "class_averaged", "a b", "tab\there", "new\nline",
    "cr\rret", 'quote"d', "single'q", "back\\slash", "comma,sep", "\x00\x01\x1f",
    "\x7f", "\x80\xa0\xff", "é日本", chr(0x2028) + chr(0x2029), chr(0xFEFF), "😀", "\U0001f600x",
    "\U0010ffff", "/ok", "<>&", "\b\f", "~ !",
]  # fmt: skip


def json_strings() -> list[dict]:
    return [{"text": s, "json": json.dumps(s)} for s in STRINGS]


CSV_ROWS = [
    ["seq", "metric", "field", "value"],
    ["toy-0001", "HOTA", "HOTA", repr(1.0)],
    ["a,b", 'c"d', "e\nf", "g\rh"],
    ["", "x", "", ""],
    [""],
    [" lead", "trail ", " ", "\t"],
    ["'", "é日本", "😀", "a\r\nb"],
    ['"', '""', "x", "nan"],
    ["class_averaged", "COMBINED", "Count", "inf"],
]


def csv_rows() -> list[dict]:
    out = []
    for row in CSV_ROWS:
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerow(row)
        out.append({"row": row, "csv": buffer.getvalue()})
    return out


PATHS = [
    "", ".", "./", "..", "a", "a/", "a//b", "a/./b", "./a", "a/..", "/", "//", "///",
    "//a", "///a", "/a/b/", "/a//b/./c/.", "missing-predictions", "a/b/../c",
    " spaced dir/", "./.", "x/./", "/./a",
]  # fmt: skip


def paths() -> list[dict]:
    return [{"text": s, "str": str(PurePosixPath(s))} for s in PATHS]


def main() -> None:
    data = {
        "format": 1,
        "python": sys.version.split()[0],
        "chunk": CHUNK,
        "sets": seeded_sets(),
        "edge": edge_cases(),
        "json_strings": json_strings(),
        "csv_rows": csv_rows(),
        "paths": paths(),
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {OUT} ({sum(s['n'] for s in data['sets'])} seeded doubles)")


if __name__ == "__main__":
    main()
