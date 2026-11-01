import json
import math
import random
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.testdata.reference import require_reference

require_reference()

from moteval.benchmarks.motchallenge import read_seq_length  # pyrefly: ignore[missing-import]

OUT = Path(__file__).resolve().parents[2] / "tests" / "integration" / "data" / "pytext"
Json = Any


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def outcome(fn) -> Json:
    try:
        return {"ok": fn()}
    except Exception as err:  # noqa: BLE001  (every exception is a verdict)
        return {"error": type(err).__name__, "message": str(err)}


SPACES = " \t\n\x0b\x0c\r\x1c\x1d\x1e\x1f\x85\xa0    　"
NUMBER_EDGES = [
    "",
    " ",
    "1",
    "-1",
    "+1",
    "1.5",
    "-0",
    "-0.0",
    "0",
    "00",
    "007",
    "1_000",
    "_1",
    "1_",
    "1__0",
    "1_0.5",
    "1._5",
    "1_.5",
    "1_e5",
    "1e1_0",
    "1e_10",
    "inf",
    "+inf",
    "-inf",
    "Inf",
    "INFINITY",
    "infinity",
    "-Infinity",
    "infinit",
    "infinityy",
    "nan",
    "NaN",
    "-nan",
    "+nan",
    "nan(1)",
    "nanx",
    ".",
    "1.",
    ".5",
    "+.5",
    "-.5",
    "e5",
    "1e",
    "1e+",
    "1e-",
    "1e5",
    "1E5",
    "1e+05",
    "1.e5",
    ".e5",
    "1..5",
    "1.5.5",
    "0x10",
    "0b1",
    "0o7",
    "1,5",
    " 1.5\t",
    "\x851.5",
    "　1.5",
    "\x1c1",
    "1\x1f",
    "\xa01",
    "١.٥",
    "٣",
    "1\x00",
    "\x00",
    "１２",
    "1e999999999999999999999",
    "-1e999999999999999999999",
    "1e-999999999999999999999",
    "4.9e-324",
    "2.4703282292062327e-324",
    "2.4703282292062328e-324",
    "1.7976931348623157e308",
    "1.7976931348623158e308",
    "1.7976931348623159e308",
    "9223372036854775807",
    "9223372036854775808",
    "-9223372036854775808",
    "-9223372036854775809",
    "18446744073709551616",
    "1" * 4300,
    "1" * 4301,
    "0" * 4301,
    "-" + "0" * 4300,
    " -0 ",
    "+-1",
    "--1",
    "- 1",
    "1 2",
    "1_2_3",
    "١_٢",
    "1 ",
    " ",
    "1\x7f",
    "\x7f",
    "1" + chr(0x200B),
    "0.1",
    "0.30000000000000004",
    "123456789012345678901234567890",
    "1e308",
    "1e309",
    "-1e-400",
    "5e-324",
    "1e-323",
    "0.000000000000000000000000000001e30",
    "100000000000000000000000e-23",
    "+",
    "-",
    "_",
    "1_١",
    "١٢٣",
    "০",
    "\U0001d7ce",
    "1̀",
]


def random_number_strings(rng: random.Random, n: int) -> list[str]:
    alphabet = list("0123456789.eE+-_ \tinfaNIx,") + ["١", " ", "\x1c", "　"]
    out = []
    for _ in range(n):
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 10))))
    for _ in range(n):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randrange(1, 26)))
        point = rng.randrange(0, len(digits) + 1)
        mantissa = digits[:point] + "." + digits[point:] if rng.random() < 0.7 else digits
        exp = f"e{rng.randrange(-340, 320)}" if rng.random() < 0.6 else ""
        sign = rng.choice(["", "-", "+"])
        out.append(sign + mantissa + exp)
    for _ in range(n):
        x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        out.append(repr(x))
        out.append(f"{x:.17g}")
    for _ in range(n // 4):
        out.append(str(rng.randrange(-(2**70), 2**70)))
    return out


def dump_numbers(rng: random.Random) -> Json:
    cases: list[Json] = []
    for s in NUMBER_EDGES + random_number_strings(rng, 500):
        f = outcome(lambda s=s: bits(float(s)))
        i = outcome(lambda s=s: int(s))
        if "ok" in i:
            value = i["ok"]
            if -(2**63) <= value < 2**63:
                i = {"ok": value}
            else:
                i = {"out_of_range": "negative" if value < 0 else "positive"}
        cases.append({"input": s, "float": f, "int": i})
    return cases


def dump_text(rng: random.Random) -> Json:
    alphabet = list("a,b ") + list(SPACES) + ["\r\n", "é", "\x00"]
    edges = [
        "",
        "\n",
        "a\n",
        "a\r\nb",
        "a\r\rb",
        "\r\n\r\n",
        "a\x0bb\x0cc\x1cd\x1de\x1ef\x1fg",
        "\x85a b ",
        "  a  b  ",
        "　x　",
    ]
    cases: list[Json] = []
    for s in edges + [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 12))) for _ in range(1500)
    ]:
        cases.append(
            {"input": s, "strip": s.strip(), "split": s.split(), "splitlines": s.splitlines()}
        )
    return cases


def dump_utf8(rng: random.Random) -> Json:
    pieces = [
        b"a",
        b"\x00",
        b"\x7f",
        b"\xc3\xa9",
        b"\xe2\x82\xac",
        b"\xf0\x9f\x98\x80",
        b"\x80",
        b"\xbf",
        b"\xc0",
        b"\xc1",
        b"\xc2",
        b"\xe0",
        b"\xe0\x80",
        b"\xe0\xa0",
        b"\xed\xa0",
        b"\xed\x9f",
        b"\xf0",
        b"\xf0\x80",
        b"\xf0\x90",
        b"\xf0\x90\x80",
        b"\xf4\x90",
        b"\xf4\x8f",
        b"\xf5",
        b"\xff",
        b"\xfe",
        b"\xe2\x28\xa1",
        b"\n",
    ]
    cases: list[Json] = []
    for _ in range(2000):
        data = b"".join(rng.choice(pieces) for _ in range(rng.randrange(0, 8)))
        cases.append(
            {"hex": data.hex(), "decode": outcome(lambda data=data: data.decode("utf-8") and None)}
        )
    return cases


def dump_utf8_chunked() -> Json:
    cases: list[Json] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "f.txt"

        def read() -> int:
            with open(path, encoding="locale") as f:
                return len(list(f))

        for size in (10, 8190, 8191, 8192, 8193, 16383, 16384, 20000):
            for bad in (
                b"\xff",
                b"\xe2\x82",
                b"\xf0\x9f\x98",
                b"\xe2\x28",
                b"\xc3",
                b"\xe2\x82\xac",
            ):
                for at in sorted(
                    {
                        0,
                        max(size - len(bad), 0),
                        max(min(8191, size - len(bad)), 0),
                        max(min(8190, size - len(bad)), 0),
                    }
                ):
                    body = bytearray(b"x" * size)
                    body[at : at + len(bad)] = bad
                    path.write_bytes(bytes(body))
                    cases.append(
                        {"size": size, "bad": bad.hex(), "at": at, "iterate": outcome(read)}
                    )
    return cases


def dump_paths() -> Json:
    paths = [
        "a//b/./c/",
        "//a",
        "///a",
        "./a",
        "",
        ".",
        "a/..",
        "/",
        "a/./",
        "x.txt",
        "/data/x/",
        "a/b/../c",
        "//",
        "///",
        "a/.hidden.txt",
        ".txt",
        "..txt",
        "a.",
        "a..b",
        "a.b.txt",
        "...",
        "a",
        "dir/000123.jpg",
        "000123.jpg",
        "-1.jpg",
        "a.b.c.",
    ]
    return [
        {"input": p, "str": str(Path(p)), "name": Path(p).name, "stem": Path(p).stem} for p in paths
    ]


def tag(value: Any) -> Json:
    if value is None:
        return {"t": "null"}
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "int", "v": str(value)}
    if isinstance(value, float):
        return {"t": "float", "v": bits(value)}
    if isinstance(value, str):
        return {"t": "str", "v": value}
    if isinstance(value, list):
        return {"t": "list", "v": [tag(v) for v in value]}
    return {"t": "dict", "v": [[k, tag(v)] for k, v in value.items()]}


def rust_cannot_hold(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return not -(2**127) <= value < 2**127
    if isinstance(value, str):
        return any(0xD800 <= ord(c) <= 0xDFFF for c in value)
    if isinstance(value, list):
        return any(rust_cannot_hold(v) for v in value)
    if isinstance(value, dict):
        return any(rust_cannot_hold(k) or rust_cannot_hold(v) for k, v in value.items())
    return False


JSON_EDGES = [
    "",
    " ",
    "{",
    "[",
    "}",
    "]",
    "{}",
    "[]",
    "{not valid json",
    '{"a": 1}',
    '{"a": 1,}',
    "[1,]",
    "[1 2]",
    '{"a" 1}',
    '{"a":}',
    "{1: 2}",
    "[,]",
    "[1,,2]",
    "nul",
    "null",
    "nulll",
    "tru",
    "true",
    "false",
    "fals",
    "NaN",
    "-NaN",
    "Infinity",
    "-Infinity",
    "-Inf",
    "-",
    "-x",
    "01",
    "1.",
    "1.e5",
    "1e",
    "1e+",
    "1e5",
    "1E-5",
    "-0",
    "-0.0",
    "0.5e-3",
    ".5",
    "+1",
    "1 ",
    " 1",
    "1 2",
    '"abc"',
    '"a',
    '"\\"',
    '"\\x"',
    '"\\u12"',
    '"\\u12g4"',
    '"\\u1234"',
    '"\\u1234',
    '"\\ud83d\\ude00"',
    '"\\ud83d\\u0041"',
    '"\\ud83dx"',
    '"a\x01b"',
    '"a\tb"',
    '"\\/\\b\\f"',
    "﻿{}",
    '{"a": [1, 2, {"b": null}]}',
    '{"a": 1, "a": 2}',
    "[1e400, -1e400]",
    "[" * 50 + "]" * 50,
    '{"x": 1}\n\n  x',
    "\n\n[1,\n2,\n]",
    '["é", "\U0001f600"]',
    "123456789012345678901234567890",
    "1" * 4301,
    "-" + "1" * 4301,
    "[" + "1" * 39 + "]",
    '{"annotations": [{"frame_id": 1, "detections": []}]}',
    '"\\ud800"',
    "[1]\x00",
]

TEMPLATES = [
    (
        '{"images": [{"id": 0, "file_name": "000000.jpg"}], "annotations": [{"id": 0, '
        '"image_id": 0, "bbox_id": 5, "bbox": [1.5, 2, 3e1, -4.25], "category_id": 0}]}'
    ),
    (
        '{"video": "v", "annotations": [{"frame_id": 1, "detections": [{"bbox": [10, 20, 60, '
        '120], "ape_id": 0}]}, {"frame_id": 2, "detections": []}]}'
    ),
    '[true, false, null, NaN, -Infinity, "s\\n\\u00e9", 0, -0.0, 1E+2]',
]


def mutate(rng: random.Random, doc: str) -> str:
    alphabet = '{}[]",:0123456789.eE+-tfnulrsaINy \t\n\r\\/u\x01'
    chars = list(doc)
    for _ in range(rng.randrange(1, 4)):
        op = rng.randrange(3)
        at = rng.randrange(len(chars) + 1)
        if op == 0 and chars:
            del chars[min(at, len(chars) - 1)]
        elif op == 1:
            chars.insert(at, rng.choice(alphabet))
        elif chars:
            chars[min(at, len(chars) - 1)] = rng.choice(alphabet)
    return "".join(chars)


def dump_json(rng: random.Random) -> Json:
    docs = JSON_EDGES + [mutate(rng, rng.choice(TEMPLATES)) for _ in range(1200)]
    cases: list[Json] = []
    for doc in docs:
        try:
            value = json.loads(doc)
        except Exception as err:  # noqa: BLE001
            cases.append({"input": doc, "error": type(err).__name__, "message": str(err)})
            continue
        if rust_cannot_hold(value):
            cases.append({"input": doc, "unsupported": True})
        else:
            cases.append({"input": doc, "ok": tag(value)})
    return cases


INI_EDGES = [
    "[Sequence]\nseqLength=5\n",
    "[Sequence]\nname=x\nimDir=img1\nframeRate=30\nseqLength=1200\nimWidth=1920\n",
    "[Sequence]\nseqLength = 5\n",
    "[Sequence]\nseqlength: 5\n",
    "[Sequence]\nSEQLENGTH=5\n",
    "[Sequence]\nseqLength=5\nseqLength=6\n",
    "[Sequence]\nseqLength=5\nSeqLength=6\n",
    "[Sequence]\n[Sequence]\n",
    "[DEFAULT]\nseqLength=7\n[Sequence]\n",
    "[DEFAULT]\nseqLength=7\n",
    "seqLength=5\n",
    "",
    "\n\n",
    "# only a comment\n",
    "; c\n[Sequence]\nseqLength=4\n",
    "[Sequence]\nseqLength=5\n  6\n",
    "[Sequence]\nseqLength=\n  5\n",
    "[Sequence]\nseqLength=5\n\n  6\n",
    "[Sequence]\nseqLength=5\n# c\n  6\n",
    "[Sequence]\nx=%(y)s\ny=5\nseqLength=%(x)s\n",
    "[Sequence]\nseqLength=%%5\n",
    "[Sequence]\nseqLength=5%\n",
    "[Sequence]\nseqLength=%(missing)s\n",
    "[Sequence]\na=%(b)s\nb=%(a)s\nseqLength=%(a)s\n",
    "[Sequence]\nseqLength=%(\n",
    "[Sequence]\nseqLength=%()s\n",
    "[Sequence]\nseqLength=%(x)d\nx=1\n",
    "[DEFAULT]\nn=8\n[Sequence]\nseqLength=%(N)s\n",
    "  [Sequence]\nseqLength=3\n",
    "[Sequence] trailing\nseqLength=3\n",
    "[]\nseqLength=3\n",
    "[a]b]\nseqLength=3\n",
    "[Sequence]\nkey without delimiter\nseqLength=3\n",
    "[Sequence]\n=5\nseqLength=3\n",
    "[Sequence]\nbad line\nother bad\nseqLength=3\n",
    "[Sequence]\r\nseqLength=9\r\n",
    "[Sequence]\rseqLength=9\r",
    "﻿[Sequence]\nseqLength=9\n",
    "[Sequence]\nseqLength=1_000\n",
    "[Sequence]\nseqLength=٣\n",
    "[Sequence]\nseqLength=0\n",
    "[Sequence]\nseqLength=5.0\n",
    "[Sequence]\nseqLength=five\n",
    "[Sequence]\nseqLength=\n",
    "[Sequence]\nseqLength=　5　\n",
    "[Sequence]\nseqLength:5=6\n",
    "[Sequence]\nseq Length=5\n",
    "[Sequence]\n seqLength=5\n",
    "[Sequence]\nseqLength=5\n[Other]\nseqLength=6\n",
    "[sequence]\nseqLength=5\n",
    "[Sequence]\nseqLength=5\x0c\n",
    "[Sequence]\nseqLength=5 ; comment\n",
    "[Sequence]\nseqLength=5 # comment\n",
    "[DEFAULT]\n[DEFAULT]\nseqLength=1\n[Sequence]\n",
    "[DEFAULT]\nseqLength=1\n[DEFAULT]\nseqLength=2\n[Sequence]\n",
    "[Sequence]\n\tseqLength=5\n",
    "[Sequence]\nseqLength=5\n\t\n  7\n",
    "[Sequence]\nnote=1\n    a\n\n    b\nseqLength=2\n",
    "[Sequence]\nseqLength=99999999999999999999\n",
]


def dump_ini(rng: random.Random) -> Json:
    texts = list(INI_EDGES)
    for _ in range(1500):
        texts.append(mutate_ini(rng, rng.choice(INI_EDGES[:3] + INI_EDGES[19:21])))
    cases: list[Json] = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, text in enumerate(texts):
            seq_dir = Path(tmp) / f"seq-{k}"
            seq_dir.mkdir()
            (seq_dir / "seqinfo.ini").write_bytes(text.encode("utf-8"))
            result = outcome(lambda seq_dir=seq_dir: read_seq_length(seq_dir))
            if "message" in result:
                result["message"] = result["message"].replace(str(seq_dir), "<SEQ>")
            if "ok" in result and not 0 <= result["ok"] < 2**63:
                result = {"unsupported": True}
            cases.append({"text": text, "result": result})
    return cases


def mutate_ini(rng: random.Random, text: str) -> str:
    alphabet = list("[]=:%()s# ;\n\t5aSeqLength") + ["\r\n", " ", "  "]
    chars = list(text)
    for _ in range(rng.randrange(1, 4)):
        at = rng.randrange(len(chars) + 1)
        op = rng.randrange(3)
        if op == 0 and chars:
            del chars[min(at, len(chars) - 1)]
        elif op == 1:
            chars.insert(at, rng.choice(alphabet))
        elif chars:
            chars[min(at, len(chars) - 1)] = rng.choice(alphabet)
    return "".join(chars)


def interesting_floats(rng: random.Random, n: int) -> list[float]:
    xs = [
        0.0,
        -0.0,
        math.inf,
        -math.inf,
        math.nan,
        0.125,
        0.375,
        2.675,
        1e300,
        -1e-300,
        5e-324,
        0.005,
        0.015,
        0.025,
        1.005,
        0.00005,
        0.00015,
        123.455,
        1e16,
        1e15,
        1e-4,
        1e-5,
        1e22,
        0.1,
        1 / 3,
        2**53,
        2**53 + 1.0,
        1.7976931348623157e308,
    ]
    for _ in range(n):
        xs.append(struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0])
        xs.append(rng.randrange(-(10**6), 10**6) / rng.choice([8, 16, 32, 64, 200, 2000, 20000]))
        xs.append(rng.uniform(-2000, 2000))
    return [float(x) for x in xs]


def dump_format(rng: random.Random) -> Json:
    return [
        {"bits": bits(x), "f2": format(x, ".2f"), "f4": format(x, ".4f"), "repr": repr(x)}
        for x in interesting_floats(rng, 1000)
    ]


def main() -> None:
    rng = random.Random(20261031)
    OUT.mkdir(parents=True, exist_ok=True)
    topics = {
        "numbers": dump_numbers(rng),
        "text": dump_text(rng),
        "utf8": dump_utf8(rng),
        "utf8_chunked": dump_utf8_chunked(),
        "paths": dump_paths(),
        "json": dump_json(rng),
        "ini": dump_ini(rng),
        "format": dump_format(rng),
    }
    for name, cases in topics.items():
        (OUT / f"{name}.json").write_text(json.dumps(cases, ensure_ascii=True) + "\n")
        print(f"{name}: {len(cases)} cases")
    print(f"CPython {sys.version.split()[0]}")


if __name__ == "__main__":
    main()
