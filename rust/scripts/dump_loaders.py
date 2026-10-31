"""Dump what Python moteval loads from each local benchmark, for the Rust loader gate.

For every (benchmark, split) of `scripts/reference_results.REFERENCE_CASES` this writes
``<out>/<benchmark>_<split>/``:

- ``pred/<seq>.txt``: the seeded perturbed predictions `reference_results.write_predictions`
  writes, the same files `dump_scenarios.py --real-data` scored for
  ``reference_benchmarks/<benchmark>_<split>``;
- ``data.bin`` and ``header.json``: the loaded `MOTDataset` (name, split, protocol, and
  every sequence's GT tracks and ignore regions) and Python's parse of each prediction
  file, as flat little-endian arrays (floats as raw `f64` bits);
- for mask benchmarks, ``geometry_<i>.json``: every mask geometry call
  `build_mask_sequence_data` makes for sequence ``i``, in order, with an FNV-1a 64 digest
  of each input mask list and the output. Rust replays them, because the RLE port lands
  separately.

`rust/moteval/tests/loaders_real.rs` loads each split with the Rust loader, reads the
prediction files with the Rust readers, builds every class view, and compares all of it
with these files and with ``reference_benchmarks``.

    MOTEVAL_DATA_ROOT=/data/tracking/external artifacts/rust-port/ref-venv/bin/python rust/scripts/dump_loaders.py \
        --out artifacts/rust-port/testdata/loaders_real --jobs 8
"""

import argparse
import json
import shutil
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
# After site-packages: `moteval` is the pure-Python reference (rust/scripts/reference.py).
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from rust.scripts.reference import require_reference

require_reference()

from moteval import Metric, Scores, SequenceData, evaluate
from moteval.benchmarks import default_data_root, load_dataset
from moteval.data import convert  # pyrefly: ignore[missing-module-attribute]
from moteval.data.model import MaskGtSequence
from moteval.formats import read_mot, read_mots
from scripts import reference_results

FORMAT_VERSION = 1
Json = Any
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def mask_digest(masks: list[Any]) -> int:
    """FNV-1a 64 over ``"{h} {w} {counts}\\n"`` for each mask, in order."""
    parts = []
    for m in masks:
        counts = m["counts"] if isinstance(m["counts"], bytes) else m["counts"].encode()
        parts.append(f"{m['size'][0]} {m['size'][1]} ".encode() + counts + b"\n")
    return fnv1a64(b"".join(parts))


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


class Blob:
    """Flat arrays packed 8-byte aligned into one ``data.bin``."""

    def __init__(self) -> None:
        self.data = bytearray()

    def add(self, dtype: str, values: Any) -> Json:
        if dtype == "u8":
            payload = bytes(values)
            length = len(payload)
        else:
            payload = np.asarray(values, dtype="<i8" if dtype == "i64" else "<f8").tobytes()
            length = len(payload) // 8
        self.data.extend(b"\0" * (-len(self.data) % 8))
        record = {"dtype": dtype, "offset": len(self.data), "len": length}
        self.data.extend(payload)
        return record


def box_rows(blob: Blob, rows: Any) -> Json:
    """`Track` rows: ``[frame, track_id, class_id]`` ints and ``[x, y, w, h, conf]`` floats."""
    ints = [v for t in rows for v in (t.frame, t.track_id, t.class_id)]
    floats = [float(v) for t in rows for v in (t.x, t.y, t.w, t.h, t.conf)]
    return {"len": len(rows), "ints": blob.add("i64", ints), "floats": blob.add("f64", floats)}


def mask_rows(blob: Blob, rows: Any) -> Json:
    """`MaskTrack` rows: five ints each, and the RLE strings with their byte lengths."""
    ints = [v for t in rows for v in (t.frame, t.track_id, t.class_id, t.img_h, t.img_w)]
    rles = [t.rle.encode() for t in rows]
    return {
        "len": len(rows),
        "ints": blob.add("i64", ints),
        "rle_lens": blob.add("i64", [len(r) for r in rles]),
        "rle_bytes": blob.add("u8", b"".join(rles)),
    }


class _NoOp(Metric):
    fields = ()

    def eval_sequence(self, data: SequenceData) -> Scores:
        return {}

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        return {}

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return {}

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return {}


def matrix(m: Any) -> Json:
    m = np.asarray(m, dtype=np.float64)
    return [m.shape[0], m.shape[1], [bits(v) for v in m.ravel()]]


def record_mask_geometry(dataset: Any, pred_dir: Path) -> list[list[Json]]:
    """Run `evaluate` and log every mask geometry call, per conversion, in order."""
    logs: list[list[Json]] = []
    originals = {
        name: getattr(convert, name)
        for name in (
            "masks_overlap",
            "merge_masks",
            "mask_ioa",
            "mask_iou",
            "build_mask_sequence_data",
        )
    }

    def logged(name: str) -> Any:
        def call(*args: Any) -> Any:
            result = originals[name](*args)
            if name == "masks_overlap":
                output: Json = bool(result)
            elif name == "merge_masks":
                counts = result["counts"]
                output = [result["size"][0], result["size"][1], counts.decode()]
            else:
                output = matrix(result)
            logs[-1].append(
                {"kind": name, "digests": [mask_digest(list(a)) for a in args], "output": output}
            )
            return result

        return call

    def build(*args: Any, **kwargs: Any) -> Any:
        logs.append([])
        return originals["build_mask_sequence_data"](*args, **kwargs)

    import moteval.eval as eval_module

    try:
        for name in ("masks_overlap", "merge_masks", "mask_ioa", "mask_iou"):
            setattr(convert, name, logged(name))
        eval_module.build_mask_sequence_data = build  # pyrefly: ignore[missing-attribute]
        evaluate(dataset, pred_dir, [_NoOp()])
    finally:
        for name, fn in originals.items():
            setattr(convert, name, fn)
        eval_module.build_mask_sequence_data = originals["build_mask_sequence_data"]  # pyrefly: ignore[missing-attribute]
    return logs


def dump_case(job: tuple[str, str, Path, Path]) -> str:
    benchmark, split, root, out = job
    dataset = load_dataset(benchmark, root=root / benchmark, split=split)
    case_dir = out / f"{benchmark}_{split}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    pred_dir = case_dir / "pred"
    pred_dir.mkdir(parents=True)
    reference_results.write_predictions(dataset, pred_dir)
    is_masks = isinstance(dataset.sequences[0], MaskGtSequence)
    multi_class = len(dataset.protocol.eval_classes) > 1
    blob = Blob()
    sequences: list[dict[str, Json]] = []
    for seq in dataset.sequences:
        pred_path = pred_dir / f"{seq.name}.txt"
        if is_masks:
            rows = mask_rows
            preds = read_mots(pred_path)
        else:
            rows = box_rows
            preds = read_mot(pred_path, class_column=multi_class)
        sequences.append(
            {
                "name": seq.name,
                "num_timesteps": seq.num_timesteps,
                "tracks": rows(blob, seq.tracks),
                "ignore_regions": rows(blob, seq.ignore_regions),
                "preds": rows(blob, preds),
            }
        )
    if is_masks:
        logs = record_mask_geometry(dataset, pred_dir)
        classes = len(dataset.protocol.eval_classes)
        assert len(logs) == classes * len(dataset.sequences), "one conversion per class view"
        for i, log in enumerate(logs):
            (case_dir / f"geometry_{i}.json").write_text(json.dumps(log, separators=(",", ":")))
    p = dataset.protocol
    header = {
        "format": FORMAT_VERSION,
        "benchmark": benchmark,
        "split": split,
        "name": dataset.name,
        "dataset_split": dataset.split,
        "kind": "masks" if is_masks else "boxes",
        "protocol": {
            "name": p.name,
            "convention": [p.frame_convention.name, p.frame_convention.first_frame],
            "eval_classes": list(p.eval_classes),
            "distractor_classes": list(p.distractor_classes),
            "distractor_iou_threshold": bits(p.distractor_iou_threshold),
            "ignore_iou_threshold": bits(p.ignore_iou_threshold),
            "drop_zero_conf_gt": p.drop_zero_conf_gt,
            "matching_fill": bits(p.matching_fill),
        },
        "sequences": sequences,
    }
    (case_dir / "data.bin").write_bytes(bytes(blob.data))
    (case_dir / "header.json").write_text(json.dumps(header, indent=1) + "\n")
    rows_total = sum(s["tracks"]["len"] for s in sequences)
    return f"{benchmark}/{split}: {len(sequences)} sequences, {rows_total} GT rows"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "artifacts/rust-port/testdata/loaders_real"
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    root = args.root if args.root is not None else default_data_root()
    args.out.mkdir(parents=True, exist_ok=True)
    work = [(b, s, root, args.out) for b, s in reference_results.REFERENCE_CASES]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(dump_case, work):
            print(line, flush=True)
    print(f"wrote {len(work)} cases to {args.out}")


if __name__ == "__main__":
    main()
