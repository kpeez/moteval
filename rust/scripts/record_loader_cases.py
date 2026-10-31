"""Record what Python moteval returns for every call the loader and data tests make.

The port gate says: for every case in ``tests/test_loaders.py`` and
``tests/test_data.py``, Rust builds the same data as Python. This script runs those two
test files under pytest with a recording plugin. The plugin wraps each ported function
(readers and writers, the loaders, ``read_seq_length``, ``preprocess_frame``,
``build_sequence_data``, ``build_mask_sequence_data``, box IoU and IoA) and the
missing-prediction check of ``evaluate``. For each call it stores:

- the inputs, with every path under the test's ``tmp_path`` written as ``<TMP>``;
- a snapshot of the files under ``tmp_path`` just before the call;
- Python's result, or its exception type and message (``<TMP>`` again).

Calls nested in another recorded call are not cases of their own, except inside
``evaluate``, which is not ported here. Inside ``build_mask_sequence_data`` the mask
geometry calls (``masks_overlap``, ``merge_masks``, ``mask_ioa``, ``mask_iou``) are
recorded in order, so Rust can replay the conversion without the RLE port.

Every test of the two files must record at least one case, or the run fails. Output:
``rust/testdata/loader_cases/<test file>/<test name>.json``, read by
``rust/moteval/tests/loader_cases.rs``.

    uv run python rust/scripts/record_loader_cases.py
"""

import functools
import hashlib
import inspect
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = _REPO_ROOT / "rust" / "testdata" / "loader_cases"
TEST_FILES = ("tests/test_loaders.py", "tests/test_data.py")
Json = Any

# Functions whose calls are cases, by module.
TARGETS: dict[str, tuple[str, ...]] = {
    "moteval.formats": ("read_mot", "read_mots", "write_mot", "write_mots"),
    "moteval.data.convert": ("build_sequence_data", "build_mask_sequence_data"),
    "moteval.data.protocol": ("preprocess_frame",),
    "moteval.data.similarity": (
        "box_iou",
        "box_ioa",
        "masks_overlap",
        "merge_masks",
        "mask_ioa",
        "mask_iou",
    ),
    "moteval.benchmarks": ("load_dataset", "default_data_root"),
    "moteval.benchmarks.motchallenge": (
        "load_layout",
        "load_motchallenge",
        "read_seq_length",
        "max_frame_seq_length",
    ),
    "moteval.benchmarks.animaltrack": ("load_animaltrack",),
    "moteval.benchmarks.bft": ("load_bft",),
    "moteval.benchmarks.chimpact": ("load_chimpact",),
    "moteval.benchmarks.dancetrack": ("load_dancetrack",),
    "moteval.benchmarks.gmot40": ("load_gmot40",),
    "moteval.benchmarks.mots20": ("load_mots20", "load_mots"),
    "moteval.benchmarks.panaf500": ("load_panaf500",),
    "moteval.benchmarks.sportsmot": ("load_sportsmot",),
    "moteval.benchmarks.uavdt": ("load_uavdt",),
    "moteval.eval": ("evaluate",),
}
MASK_GEOMETRY = ("masks_overlap", "merge_masks", "mask_ioa", "mask_iou")


# ------------------------------------------------------------------------- encoding


def bits(x: Any) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


class Context:
    """The running test: its id, its ``tmp_path``, and the cases recorded so far."""

    def __init__(self, nodeid: str, tmp: Path | None) -> None:
        self.nodeid = nodeid
        self.tmp = tmp
        self.cases: list[Json] = []
        self.snapshots: dict[str, Json] = {}

    def token(self, text: str) -> str:
        return text.replace(str(self.tmp), "<TMP>") if self.tmp is not None else text

    def path(self, p: Any) -> str:
        return self.token(str(Path(p)))

    def snapshot(self) -> str | None:
        """Store the files under ``tmp_path`` (deduplicated); return the snapshot id."""
        if self.tmp is None:
            return None
        entries = []
        for dirpath, dirnames, filenames in os.walk(self.tmp):
            dirnames.sort()
            rel_dir = Path(dirpath).relative_to(self.tmp)
            for d in dirnames:
                full = Path(dirpath) / d
                assert not full.is_symlink(), f"symlink in a test tree: {full}"
                entries.append({"path": str(rel_dir / d), "dir": True})
            for f in sorted(filenames):
                full = Path(dirpath) / f
                assert not full.is_symlink(), f"symlink in a test tree: {full}"
                data = full.read_bytes()
                try:
                    entries.append({"path": str(rel_dir / f), "text": data.decode("utf-8")})
                except UnicodeDecodeError:
                    entries.append({"path": str(rel_dir / f), "hex": data.hex()})
        key = hashlib.sha1(json.dumps(entries).encode()).hexdigest()[:12]
        self.snapshots.setdefault(key, entries)
        return key


def track(t: Any) -> Json:
    return [t.frame, t.track_id, t.class_id, *(bits(v) for v in (t.x, t.y, t.w, t.h, t.conf))]


def mask_track(t: Any) -> Json:
    return [t.frame, t.track_id, t.class_id, t.img_h, t.img_w, t.rle]


def protocol(p: Any) -> Json:
    return {
        "name": p.name,
        "convention": [p.frame_convention.name, p.frame_convention.first_frame],
        "eval_classes": list(p.eval_classes),
        "distractor_classes": list(p.distractor_classes),
        "distractor_iou_threshold": bits(p.distractor_iou_threshold),
        "ignore_iou_threshold": bits(p.ignore_iou_threshold),
        "drop_zero_conf_gt": p.drop_zero_conf_gt,
        "matching_fill": bits(p.matching_fill),
    }


def sequence(s: Any) -> Json:
    from moteval.data.model import MaskGtSequence

    rows = mask_track if isinstance(s, MaskGtSequence) else track
    return {
        "kind": "masks" if isinstance(s, MaskGtSequence) else "boxes",
        "name": s.name,
        "num_timesteps": s.num_timesteps,
        "tracks": [rows(t) for t in s.tracks],
        "ignore_regions": [rows(t) for t in s.ignore_regions],
    }


def dataset(d: Any) -> Json:
    return {
        "name": d.name,
        "split": d.split,
        "protocol": protocol(d.protocol),
        "sequences": [sequence(s) for s in d.sequences],
    }


def matrix(m: Any) -> Json:
    m = np.asarray(m, dtype=np.float64)
    assert m.ndim == 2, m.shape
    return [m.shape[0], m.shape[1], [bits(v) for v in m.ravel()]]


def rle(r: Any) -> Json:
    counts = r["counts"]
    return [r["size"][0], r["size"][1], counts.decode() if isinstance(counts, bytes) else counts]


def dets(arr: Any) -> Json:
    """A frame's detections: boxes as bits, masks as ``[h, w, counts]``."""
    arr = np.asarray(arr)
    if arr.dtype == object:
        return {"masks": [rle(r) for r in arr]}
    return {
        "boxes": [
            [bits(v) for v in row] for row in np.asarray(arr, dtype=np.float64).reshape(-1, 4)
        ]
    }


def ints(arr: Any) -> list[int]:
    return [int(v) for v in np.asarray(arr).ravel()]


def floats(arr: Any) -> list[int]:
    return [bits(v) for v in np.asarray(arr, dtype=np.float64).ravel()]


def sequence_data(d: Any) -> Json:
    return {
        "name": d.name,
        "num_timesteps": d.num_timesteps,
        "num_gt_ids": d.num_gt_ids,
        "num_pred_ids": d.num_pred_ids,
        "num_gt_dets": d.num_gt_dets,
        "num_pred_dets": d.num_pred_dets,
        "gt_ids": [ints(f) for f in d.gt_ids],
        "pred_ids": [ints(f) for f in d.pred_ids],
        "pred_confidences": [floats(f) for f in d.pred_confidences],
        "gt_dets": [dets(f) for f in d.geometry.gt],
        "pred_dets": [dets(f) for f in d.geometry.pred],
        "similarity": [matrix(m) for m in d.similarity],
    }


def raw_frame(f: Any) -> Json:
    return {
        "gt_ids": ints(f.gt_ids),
        "gt_classes": ints(f.gt_classes),
        "gt_conf": floats(f.gt_conf),
        "gt_dets": dets(f.gt_dets),
        "pred_ids": ints(f.pred_ids),
        "pred_classes": ints(f.pred_classes),
        "pred_confidences": floats(f.pred_confidences),
        "pred_dets": dets(f.pred_dets),
        "similarity": matrix(f.similarity),
        "ignore_ioa": matrix(f.ignore_ioa),
    }


def preprocessed_frame(f: Any) -> Json:
    return {
        "gt_ids": ints(f.gt_ids),
        "gt_dets": dets(f.gt_dets),
        "pred_ids": ints(f.pred_ids),
        "pred_confidences": floats(f.pred_confidences),
        "pred_dets": dets(f.pred_dets),
        "similarity": matrix(f.similarity),
    }


def mc_config(ctx: Context, config: Any) -> Json:
    from moteval.benchmarks import motchallenge as mc

    defaults = (
        mc._default_seq_names,
        mc._default_gt_path,
        mc._seqinfo_seq_length,
        mc._default_ignore_path,
    )
    hooks = (config.seq_names, config.gt_path, config.seq_length, config.ignore_path)
    if hooks != defaults:
        raise AssertionError(f"{ctx.nodeid}: load_layout with custom hooks is not recorded")
    return {"name": config.name, "protocol": protocol(config.protocol), "class_id": config.class_id}


def encode_inputs(ctx: Context, kind: str, args: dict[str, Any]) -> Json:
    p = ctx.path
    if kind in ("read_mot", "read_mots"):
        out = {"path": p(args["path"])}
        if kind == "read_mot":
            out["class_column"] = args["class_column"]
        return out
    if kind in ("write_mot", "write_mots"):
        rows = track if kind == "write_mot" else mask_track
        return {"path": p(args["path"]), "tracks": [rows(t) for t in args["tracks"]]}
    if kind in ("build_sequence_data", "build_mask_sequence_data"):
        rows = track if kind == "build_sequence_data" else mask_track
        return {
            "gt": sequence(args["gt"]),
            "pred_tracks": [rows(t) for t in args["pred_tracks"]],
            "protocol": protocol(args["protocol"]),
            "cls_id": args["cls_id"],
        }
    if kind == "preprocess_frame":
        return {
            "frame": raw_frame(args["frame"]),
            "protocol": protocol(args["protocol"]),
            "cls_id": args["cls_id"],
        }
    if kind in ("box_iou", "box_ioa"):
        return {"a": dets(args["boxes_a"]), "b": dets(args["boxes_b"])}
    if kind == "default_data_root":
        return {"env": env_value(ctx)}
    if kind == "load_dataset":
        root = args["root"]
        return {
            "name": args["name"],
            "root": None if root is None else p(root),
            "split": args["split"],
            "env": env_value(ctx),
        }
    if kind == "load_layout":
        return {
            "config": mc_config(ctx, args["config"]),
            "root": p(args["root"]),
            "split": args["split"],
        }
    if kind == "read_seq_length":
        return {"seq_dir": p(args["seq_dir"])}
    if kind == "max_frame_seq_length":
        return {
            "seq_name": args["seq_name"],
            "tracks": [track(t) for t in args["tracks"]],
            "offset": args["offset"],
        }
    if kind == "load_mots":
        return {"root": p(args["root"]), "split": args["split"], "class_id": args["class_id"]}
    if kind.startswith("load_"):
        return {"root": p(args["root"]), "split": args["split"]}
    raise AssertionError(kind)


def env_value(ctx: Context) -> str | None:
    value = os.environ.get("MOTEVAL_DATA_ROOT")
    return None if value is None else ctx.token(value)


def encode_result(ctx: Context, kind: str, result: Any, args: dict[str, Any]) -> Json:
    if kind in ("read_mot", "read_mots"):
        rows = track if kind == "read_mot" else mask_track
        return [rows(t) for t in result]
    if kind in ("write_mot", "write_mots"):
        return Path(args["path"]).read_text()
    if kind in ("build_sequence_data", "build_mask_sequence_data"):
        return sequence_data(result)
    if kind == "preprocess_frame":
        return preprocessed_frame(result)
    if kind in ("box_iou", "box_ioa"):
        return matrix(result)
    if kind == "default_data_root":
        return ctx.path(result)
    if kind in ("read_seq_length", "max_frame_seq_length"):
        return int(result)
    return dataset(result)


# -------------------------------------------------------------------------- recording

STATE: dict[str, Any] = {"ctx": None, "stack": [], "geometry": None}
RESULTS: dict[str, Context] = {}


def wrap(kind: str, fn: Any) -> Any:
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx: Context | None = STATE["ctx"]
        stack: list[str] = STATE["stack"]
        if ctx is None:
            return fn(*args, **kwargs)
        if kind in MASK_GEOMETRY:
            return record_geometry(kind, fn, args, kwargs)
        if kind == "evaluate":
            record_missing(ctx, signature.bind(*args, **kwargs).arguments)
            stack.append(kind)
            try:
                return fn(*args, **kwargs)
            finally:
                stack.pop()
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        record = not [k for k in stack if k != "evaluate"]
        if not record:
            stack.append(kind)
            try:
                return fn(*args, **kwargs)
            finally:
                stack.pop()
        case: Json = {
            "kind": kind,
            "snapshot": ctx.snapshot(),
            "inputs": encode_inputs(ctx, kind, bound.arguments),
        }
        outer_geometry = STATE["geometry"]
        if kind == "build_mask_sequence_data":
            STATE["geometry"] = case.setdefault("geometry", [])
        stack.append(kind)
        try:
            result = fn(*args, **kwargs)
        except Exception as err:
            case["error"] = {"type": type(err).__name__, "message": ctx.token(str(err))}
            ctx.cases.append(case)
            raise
        finally:
            stack.pop()
            STATE["geometry"] = outer_geometry
        case["ok"] = encode_result(ctx, kind, result, bound.arguments)
        ctx.cases.append(case)
        return result

    return wrapper


def record_geometry(kind: str, fn: Any, args: Any, kwargs: Any) -> Any:
    """A mask geometry call inside a recorded mask conversion: inputs and output."""
    log = STATE["geometry"]
    result = fn(*args, **kwargs)
    if log is not None:
        inputs = [[rle(r) for r in masks] for masks in args]
        if kind == "masks_overlap":
            output: Json = bool(result)
        elif kind == "merge_masks":
            output = rle(result)
        else:
            output = matrix(result)
        log.append({"kind": kind, "inputs": inputs, "output": output})
    return result


def record_missing(ctx: Context, args: dict[str, Any]) -> None:
    """`evaluate`'s missing-prediction check, as Python computes it, before the call."""
    dataset_arg = args["dataset"]
    pred_dir = Path(args["predictions"])
    names = [s.name for s in dataset_arg.sequences]
    missing = [n for n in names if not (pred_dir / f"{n}.txt").is_file()]
    case: Json = {
        "kind": "missing_predictions",
        "snapshot": ctx.snapshot(),
        "inputs": {"pred_dir": ctx.path(pred_dir), "names": names},
        "ok": missing,
    }
    if missing:
        case["message"] = ctx.token(
            f"prediction directory {pred_dir} has no <seq>.txt file for {len(missing)} "
            f"sequence(s): {', '.join(missing)} (an empty file means no predictions)"
        )
    ctx.cases.append(case)


ORIGINALS: dict[int, Any] = {}


def patch_everything() -> None:
    """Replace every binding of a target function, in every loaded module, by its wrapper."""
    import importlib

    for module_name, names in TARGETS.items():
        module = importlib.import_module(module_name)
        for name in names:
            original = getattr(module, name)
            original = getattr(original, "__wrapped__", original)
            ORIGINALS.setdefault(id(original), (name, original, wrap(name, original)))
    by_id = {key: wrapper for key, (_, _, wrapper) in ORIGINALS.items()}
    for module in list(sys.modules.values()):
        mod_name = getattr(module, "__name__", "")
        if not mod_name.startswith(("moteval", "tests")):
            continue
        for attr, value in list(vars(module).items()):
            if id(value) in by_id and callable(value):
                setattr(module, attr, by_id[id(value)])
    from moteval.benchmarks import BENCHMARKS

    for key, value in list(BENCHMARKS.items()):
        if id(value) in by_id:
            BENCHMARKS[key] = by_id[id(value)]


class Recorder:
    """The pytest plugin."""

    def pytest_configure(self, config: Any) -> None:
        patch_everything()

    def pytest_collection_modifyitems(self, session: Any, config: Any, items: list[Any]) -> None:
        patch_everything()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: Any) -> Any:
        tmp = item.funcargs.get("tmp_path")
        ctx = Context(item.nodeid, Path(tmp) if tmp is not None else None)
        STATE["ctx"] = ctx
        try:
            yield
        finally:
            STATE["ctx"] = None
            STATE["stack"].clear()
            RESULTS[item.nodeid] = ctx


def slug(nodeid: str) -> str:
    """A file name for a test: its readable name plus a hash, since parameter ids such as
    ``[]`` and ``[   ]`` collapse to the same readable name."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid.split("::", 1)[1]).strip("_")
    return f"{name}-{hashlib.sha1(nodeid.encode()).hexdigest()[:8]}"


def main() -> None:
    recorder = Recorder()
    os.chdir(_REPO_ROOT)
    code = pytest.main(["-q", "-p", "no:cacheprovider", *TEST_FILES], plugins=[recorder])
    if code != 0:
        raise SystemExit(f"the recorded tests failed (pytest exit {code})")
    if OUT.exists():
        shutil.rmtree(OUT)
    empty = [nodeid for nodeid, ctx in RESULTS.items() if not ctx.cases]
    if empty:
        raise SystemExit("tests that recorded no case:\n  " + "\n  ".join(empty))
    total = 0
    for nodeid, ctx in sorted(RESULTS.items()):
        test_file = Path(nodeid.split("::", 1)[0]).stem
        path = OUT / test_file / f"{slug(nodeid)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"test": nodeid, "snapshots": ctx.snapshots, "cases": ctx.cases}
        path.write_text(json.dumps(body, separators=(",", ":")) + "\n")
        total += len(ctx.cases)
    kinds: dict[str, int] = {}
    for ctx in RESULTS.values():
        for case in ctx.cases:
            kinds[case["kind"]] = kinds.get(case["kind"], 0) + 1
    print(f"recorded {total} cases from {len(RESULTS)} tests into {OUT}")
    for kind, n in sorted(kinds.items()):
        print(f"  {kind}: {n}")


if __name__ == "__main__":
    main()
