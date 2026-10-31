"""Dump each parity scenario's `SequenceData` inputs and expected scores for the Rust tests.

Synthetic scenarios (`tests/scenarios.py`, oracle numbers in `tests/fixtures/synthetic_*.json`)
go to ``rust/testdata/scenarios/<fixture>/<case>.json``, one self-contained JSON file per
case, committed. With ``--real-data`` the script also dumps the real-data cases
(`tests/fixtures/real_data.json` and the nine-benchmark `reference_benchmarks.json`) to
``--real-out`` (default ``artifacts/rust-port/testdata``): one ``case.json`` per case plus a
small JSON header and a flat little-endian ``.bin`` file per sequence. They are too large to
commit. ``rust/testdata/README.md`` documents both layouts.

Each sequence is captured exactly as `moteval.evaluate` builds it, through a recording
metric, so the Rust metrics see the same frame-major input as the Python metrics. Every
case also stores Python moteval's own scores, and the script checks that they equal the
fixture bit for bit before it writes anything, so a dumped case cannot disagree with its
oracle.

    uv run python rust/scripts/dump_scenarios.py
    MOTEVAL_DATA_ROOT=/data/tracking/external uv run python rust/scripts/dump_scenarios.py \
        --real-data --jobs 8
"""

import argparse
import itertools
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moteval import (
    CLEAR,
    HOTA,
    Count,
    Identity,
    JAndF,
    Metric,
    MOTDataset,
    Scores,
    SequenceData,
    TrackMAP,
    evaluate,
)
from moteval.benchmarks import default_data_root, load_dataset
from moteval.data.model import BoxGeometry
from scripts import reference_results
from tests import scenarios

SYNTHETIC_OUT = Path(__file__).resolve().parents[1] / "testdata" / "scenarios"
REAL_OUT = _REPO_ROOT / "artifacts" / "rust-port" / "testdata"
FIXTURES = _REPO_ROOT / "tests" / "fixtures"
FORMAT_VERSION = 1

METRIC_TYPES: dict[str, type[Metric]] = {
    "HOTA": HOTA,
    "CLEAR": CLEAR,
    "Identity": Identity,
    "Count": Count,
    "JAndF": JAndF,
    "TrackMAP": TrackMAP,
}
BOX_METRICS = ("HOTA", "CLEAR", "Identity", "Count")
MOTS_METRICS = (*BOX_METRICS, "JAndF")

Json = Any


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_value(value: float | np.ndarray) -> Json:
    """A score value as its numpy shape plus each element's float64 bits."""
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return {"shape": list(array.shape), "bits": array.view(np.uint64).ravel().tolist()}


def encode_scores(scores: Scores) -> Json:
    """Scores as ``[[field, value], ...]`` in insertion order (the export order)."""
    return [[field, encode_value(value)] for field, value in scores.items()]


def encode_metric_scores(by_metric: dict[str, Scores]) -> Json:
    return [[name, encode_scores(scores)] for name, scores in by_metric.items()]


def _concat(parts: list[np.ndarray], dtype: str) -> np.ndarray:
    flat = [np.asarray(p, dtype=dtype).ravel() for p in parts]
    return np.concatenate(flat) if flat else np.zeros(0, dtype=dtype)


def flat_arrays(data: SequenceData) -> dict[str, tuple[str, np.ndarray | bytes]]:
    """The frame-major fields as flat arrays: ``name -> (dtype, values)``.

    Per-frame ragged fields are concatenated over frames; ``gt_counts`` and
    ``pred_counts`` give each frame's length, and each frame's similarity is its
    ``gt x pred`` matrix in C order.
    """
    arrays: dict[str, tuple[str, np.ndarray | bytes]] = {
        "gt_counts": ("i64", np.array([len(ids) for ids in data.gt_ids], dtype="<i8")),
        "pred_counts": ("i64", np.array([len(ids) for ids in data.pred_ids], dtype="<i8")),
        "gt_ids": ("i64", _concat(list(data.gt_ids), "<i8")),
        "pred_ids": ("i64", _concat(list(data.pred_ids), "<i8")),
        "pred_confidences": ("f64", _concat(list(data.pred_confidences), "<f8")),
        "similarity": ("f64", _concat(list(data.similarity), "<f8")),
    }
    for t, sim in enumerate(data.similarity):
        assert sim.shape == (len(data.gt_ids[t]), len(data.pred_ids[t])), (data.name, t)
    if isinstance(data.geometry, BoxGeometry):
        arrays["gt_boxes"] = ("f64", _concat(list(data.geometry.gt), "<f8"))
        arrays["pred_boxes"] = ("f64", _concat(list(data.geometry.pred), "<f8"))
        return arrays
    for side, frames in (("gt", data.geometry.gt), ("pred", data.geometry.pred)):
        rles = [rle for frame in frames for rle in frame]
        counts = [rle["counts"] for rle in rles]
        assert all(isinstance(c, bytes) for c in counts), "mask counts must be bytes"
        arrays[f"{side}_mask_heights"] = ("i64", np.array([r["size"][0] for r in rles], "<i8"))
        arrays[f"{side}_mask_widths"] = ("i64", np.array([r["size"][1] for r in rles], "<i8"))
        arrays[f"{side}_mask_count_lens"] = ("i64", np.array([len(c) for c in counts], "<i8"))
        arrays[f"{side}_mask_counts"] = ("u8", b"".join(counts))
    return arrays


def sequence_meta(key: str, data: SequenceData) -> Json:
    return {
        "key": key,
        "name": data.name,
        "num_timesteps": data.num_timesteps,
        "num_gt_ids": data.num_gt_ids,
        "num_pred_ids": data.num_pred_ids,
        "num_gt_dets": data.num_gt_dets,
        "num_pred_dets": data.num_pred_dets,
        "geometry": "boxes" if isinstance(data.geometry, BoxGeometry) else "masks",
    }


def inline_sequence(key: str, data: SequenceData) -> Json:
    """A sequence record with its arrays inline: floats as bits, mask counts as ASCII."""
    record = sequence_meta(key, data)
    arrays = {}
    for name, (dtype, values) in flat_arrays(data).items():
        if dtype == "u8":
            assert isinstance(values, bytes)
            arrays[name] = {"dtype": "ascii", "data": values.decode("ascii")}
        elif dtype == "f64":
            assert isinstance(values, np.ndarray)
            arrays[name] = {"dtype": "f64", "data": values.view(np.uint64).tolist()}
        else:
            assert isinstance(values, np.ndarray)
            arrays[name] = {"dtype": "i64", "data": values.tolist()}
    record["arrays"] = arrays
    return record


def binary_sequence(directory: Path, stem: str, key: str, data: SequenceData) -> Json:
    """Write ``<stem>.bin`` and ``<stem>.json`` (its header); return the case's reference."""
    record = sequence_meta(key, data)
    arrays = {}
    blob = bytearray()
    for name, (dtype, values) in flat_arrays(data).items():
        payload = values if isinstance(values, bytes) else values.tobytes()
        blob.extend(b"\0" * (-len(blob) % 8))  # every array starts 8-byte aligned
        length = len(payload) if dtype == "u8" else len(payload) // 8
        arrays[name] = {"dtype": dtype, "offset": len(blob), "len": length}
        blob.extend(payload)
    record["bin"] = f"{stem}.bin"
    record["arrays"] = arrays
    (directory / f"{stem}.bin").write_bytes(bytes(blob))
    (directory / f"{stem}.json").write_text(json.dumps(record, indent=1) + "\n")
    return {"header": f"{stem}.json"}


# ---------------------------------------------------------------------------
# Capture and scoring
# ---------------------------------------------------------------------------


class _Recorder(Metric):
    """A metric that hands every `SequenceData` that `evaluate` builds to a callback."""

    fields = ()

    def __init__(self, sink: Callable[[SequenceData], None]) -> None:
        self.sink = sink

    def eval_sequence(self, data: SequenceData) -> Scores:
        self.sink(data)
        return {}

    def combine_sequences(self, all_res: dict[str, Scores]) -> Scores:
        return {}

    def combine_classes_class_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return {}

    def combine_classes_det_averaged(self, all_res: dict[str, Scores]) -> Scores:
        return {}


class View:
    """One class view: its sequence records and Python moteval's per-sequence scores."""

    def __init__(self, cls: int | None) -> None:
        self.cls = cls
        self.records: list[Json] = []
        self.per_sequence: list[tuple[str, dict[str, Scores]]] = []


SequenceWriter = Callable[[int, int, str, SequenceData], Json]


class Capture:
    """Records sequences view by view and scores each with the case's metrics."""

    def __init__(self, metric_names: tuple[str, ...], writer: SequenceWriter) -> None:
        self.metrics = {name: METRIC_TYPES[name]() for name in metric_names}
        self.writer = writer
        self.views: list[View] = []

    def add(self, view: View, key: str, data: SequenceData) -> None:
        index = len(view.records)
        view.records.append(self.writer(len(self.views) - 1, index, key, data))
        scores = {name: metric.eval_sequence(data) for name, metric in self.metrics.items()}
        view.per_sequence.append((key, scores))

    def from_dataset(self, dataset: MOTDataset, pred_dir: Path) -> None:
        """Capture every class view exactly as `evaluate` builds it."""
        keys = [seq.name for seq in dataset.sequences]
        classes = dataset.protocol.eval_classes
        counter = itertools.count()

        def sink(data: SequenceData) -> None:
            k = next(counter)
            if k % len(keys) == 0:
                self.views.append(View(classes[k // len(keys)]))
            assert data.name == keys[k % len(keys)], (data.name, keys[k % len(keys)])
            self.add(self.views[-1], keys[k % len(keys)], data)

        evaluate(dataset, pred_dir, [_Recorder(sink)])
        assert len(self.views) == len(classes), "evaluate scored an unexpected class count"


def declared(metric: Metric, scores: Scores) -> Scores:
    return {field: value for field, value in scores.items() if field in metric.fields}


def score(capture: Capture, procedure: str) -> dict[str, Any]:
    """Python moteval's results for a captured case, by its procedure.

    ``per_class``: combine each view's sequences; with several views, also combine the
    views' raw combined state class-averaged and det-averaged, in class order.
    ``sequences_as_classes``: one view whose declared per-sequence scores are combined
    class-averaged, as ``tests/test_parity.py`` checks the HOTA class combiner.
    """
    out: dict[str, Any] = {"combined": [], "class_averaged": {}, "det_averaged": {}}
    metrics = capture.metrics
    for view in capture.views:
        combined = {}
        for name, metric in metrics.items():
            by_seq = {key: scores[name] for key, scores in view.per_sequence}
            combined[name] = metric.combine_sequences(by_seq)
        out["combined"].append(combined)
    if procedure == "per_class" and len(capture.views) > 1:
        for name, metric in metrics.items():
            raw = {str(v.cls): comb[name] for v, comb in zip(capture.views, out["combined"])}
            out["class_averaged"][name] = metric.combine_classes_class_averaged(raw)
            out["det_averaged"][name] = metric.combine_classes_det_averaged(raw)
    elif procedure == "sequences_as_classes":
        (view,) = capture.views
        for name, metric in metrics.items():
            per_seq = {key: declared(metric, scores[name]) for key, scores in view.per_sequence}
            out["class_averaged"][name] = metric.combine_classes_class_averaged(per_seq)
    return out


def check_against_oracle(label: str, oracle: dict[str, Json], ours: dict[str, Scores]) -> int:
    """Assert every oracle field equals Python moteval's value bit for bit."""
    checked = 0
    for metric, fields in oracle.items():
        for field, expected in fields.items():
            got = np.asarray(ours[metric][field], dtype=np.float64)
            want = np.asarray(expected, dtype=np.float64)
            same_nan = np.isnan(got) & np.isnan(want)
            same_bits = got.view(np.uint64) == want.view(np.uint64)
            if got.shape != want.shape or not np.all(same_bits | same_nan):
                raise AssertionError(f"{label}: {metric}.{field}: {got} != {want}")
            checked += 1
    return checked


def build_case(
    fixture: str,
    case: str,
    procedure: str,
    capture: Capture,
    oracle: dict[str, dict[str, Json]],
    *,
    raw_per_sequence: bool,
    protocol: Json,
) -> Json:
    """Check Python moteval against the oracle, then assemble the case JSON."""
    results = score(capture, procedure)
    checked = 0
    for section, expected in oracle.items():
        if section == "combined":
            (combined,) = results["combined"]
            checked += check_against_oracle(f"{fixture}/{case}", expected, combined)
        else:
            checked += check_against_oracle(
                f"{fixture}/{case} {section}", expected, results[section]
            )
    assert checked > 0, f"{fixture}/{case}: no oracle fields checked"

    def per_seq(view: View) -> Json:
        rows = []
        for key, scores in view.per_sequence:
            by_metric = {
                name: scores[name] if raw_per_sequence else declared(metric, scores[name])
                for name, metric in capture.metrics.items()
            }
            rows.append([key, encode_metric_scores(by_metric)])
        return rows

    return {
        "format": FORMAT_VERSION,
        "fixture": fixture,
        "case": case,
        "procedure": procedure,
        "protocol": protocol,
        "metrics": list(capture.metrics),
        "views": [{"class": v.cls, "sequences": v.records} for v in capture.views],
        "oracle": {
            section: [
                [metric, [[f, encode_value(v)] for f, v in fields.items()]]
                for metric, fields in expected.items()
            ]
            for section, expected in oracle.items()
        },
        "moteval": {
            "views": [
                {
                    "class": v.cls,
                    "per_sequence": per_seq(v),
                    "combined": encode_metric_scores(comb),
                }
                for v, comb in zip(capture.views, results["combined"])
            ],
            "class_averaged": encode_metric_scores(results["class_averaged"]),
            "det_averaged": encode_metric_scores(results["det_averaged"]),
        },
        "oracle_fields_checked": checked,
    }


def protocol_json(dataset: MOTDataset | None) -> Json:
    if dataset is None:
        return None
    p = dataset.protocol
    return {
        "name": p.name,
        "first_frame": p.frame_convention.first_frame,
        "eval_classes": list(p.eval_classes),
        "distractor_classes": list(p.distractor_classes),
        "matching_fill": p.matching_fill,
        "dataset": dataset.name,
        "split": dataset.split,
    }


# ---------------------------------------------------------------------------
# Synthetic cases
# ---------------------------------------------------------------------------


def _inline_writer(_view: int, _index: int, key: str, data: SequenceData) -> Json:
    return inline_sequence(key, data)


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def synthetic_cases() -> list[tuple[str, str, Json]]:
    box = _load_fixture("synthetic_box")
    mots = _load_fixture("synthetic_mots")
    trackmap = _load_fixture("synthetic_trackmap")
    cases: list[tuple[str, str, Json]] = []

    def dataset_case(
        fixture: str,
        case: str,
        procedure: str,
        metric_names: tuple[str, ...],
        oracle: dict[str, Json],
        prepare: Callable[[Path], MOTDataset],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = prepare(Path(tmp))
            capture = Capture(metric_names, _inline_writer)
            capture.from_dataset(dataset, scenarios.predictions_dir(Path(tmp)))
        body = build_case(
            fixture,
            case,
            procedure,
            capture,
            oracle,
            raw_per_sequence=True,
            protocol=protocol_json(dataset),
        )
        cases.append((fixture, case, body))

    def box_dataset(scenario: scenarios.Scenario) -> Callable[[Path], MOTDataset]:
        def prepare(tmp: Path) -> MOTDataset:
            scenarios.write_mot_scenario(tmp, scenario)
            return scenarios.build_box_dataset(tmp, scenario)

        return prepare

    def mots_dataset(scenario: scenarios.Scenario) -> Callable[[Path], MOTDataset]:
        def prepare(tmp: Path) -> MOTDataset:
            scenarios.write_mots_scenario(tmp, scenario)
            return scenarios.build_mots_dataset(tmp, scenario)

        return prepare

    for scenario in scenarios.BOX_SCENARIOS:
        oracle = {"combined": box[scenario.name]}
        dataset_case(
            "synthetic_box", scenario.name, "per_class", BOX_METRICS, oracle, box_dataset(scenario)
        )

    combine = next(
        s for s in scenarios.BOX_SCENARIOS if s.name == scenarios.COMBINE_CLASSES_SCENARIO
    )
    oracle = {"class_averaged": box["combine_classes_class_averaged"]}
    dataset_case(
        "synthetic_box",
        "combine_classes_class_averaged",
        "sequences_as_classes",
        ("HOTA",),
        oracle,
        box_dataset(combine),
    )

    def multi_class_box(tmp: Path) -> MOTDataset:
        scenarios.write_multi_class_box_scenario(tmp, combine, scenarios.sequence_classes(combine))
        return scenarios.build_multi_class_box_dataset(tmp, combine)

    oracle = {
        "class_averaged": box["multi_class_class_averaged"],
        "det_averaged": box["multi_class_det_averaged"],
    }
    dataset_case(
        "synthetic_box",
        "multi_class",
        "per_class",
        (*BOX_METRICS, "TrackMAP"),
        oracle,
        multi_class_box,
    )

    mots_scenarios = scenarios.build_mots_scenarios()
    for scenario in mots_scenarios:
        oracle = {"combined": mots[scenario.name]}
        dataset_case(
            "synthetic_mots",
            scenario.name,
            "per_class",
            MOTS_METRICS,
            oracle,
            mots_dataset(scenario),
        )

    mots_multi = next(s for s in mots_scenarios if s.name == scenarios.MOTS_MULTI_CLASS_SCENARIO)

    def multi_class_mots(tmp: Path) -> MOTDataset:
        scenarios.write_multi_class_mots_scenario(tmp, mots_multi, scenarios.mots_track_classes)
        return scenarios.build_multi_class_mots_dataset(tmp, mots_multi)

    oracle = {
        "class_averaged": mots["multi_class_class_averaged"],
        "det_averaged": mots["multi_class_det_averaged"],
    }
    dataset_case(
        "synthetic_mots", "multi_class", "per_class", MOTS_METRICS, oracle, multi_class_mots
    )

    for name, sequences in sorted(scenarios.TRACKMAP_SCENARIOS.items()):
        capture = Capture(("TrackMAP",), _inline_writer)
        view = View(None)
        capture.views.append(view)
        for key, (num_timesteps, gt_tracks, pred_tracks) in sequences.items():
            data = scenarios.build_trackmap_sequence_data(num_timesteps, gt_tracks, pred_tracks)
            capture.add(view, key, data)
        body = build_case(
            "synthetic_trackmap",
            name,
            "per_class",
            capture,
            {"combined": trackmap[name]},
            raw_per_sequence=True,
            protocol=None,
        )
        cases.append(("synthetic_trackmap", name, body))

    covered = {
        "synthetic_box": set(box),
        "synthetic_mots": set(mots),
        "synthetic_trackmap": set(trackmap),
    }
    dumped = {(f, c) for f, c, _ in cases}
    for fixture, entries in covered.items():
        for entry in entries:
            case = "multi_class" if entry.startswith("multi_class_") else entry
            assert (fixture, case) in dumped, f"fixture entry {fixture}/{entry} not dumped"
    return cases


def write_synthetic(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    cases = synthetic_cases()
    for fixture, case, body in cases:
        path = out / fixture / f"{case}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, separators=(",", ":")) + "\n")
    total = sum(p.stat().st_size for p in out.rglob("*.json"))
    print(f"wrote {len(cases)} synthetic cases to {out} ({total} bytes)", flush=True)


# ---------------------------------------------------------------------------
# Real-data cases
# ---------------------------------------------------------------------------

REAL_DATA_CASES = ("dancetrack_val", "sportsmot_val", "mots20")


def _real_data_case(case: str, root: Path, out: Path) -> str:
    """One `real_data.json` entry: the same inputs as `tests/test_parity_real.py`."""
    prepare = {
        "dancetrack_val": scenarios.prepare_dancetrack_val,
        "sportsmot_val": scenarios.prepare_sportsmot_val,
        "mots20": scenarios.prepare_mots20_sequence,
    }[case]
    metric_names = MOTS_METRICS if case == "mots20" else BOX_METRICS
    oracle = {"combined": _load_fixture("real_data")[case]}
    with tempfile.TemporaryDirectory() as tmp:
        dataset, _, _ = prepare(Path(tmp))
        return _write_real_case(
            "real_data",
            case,
            dataset,
            scenarios.predictions_dir(Path(tmp)),
            metric_names,
            oracle,
            out,
        )


def _reference_case(benchmark: str, split: str, root: Path, out: Path) -> str:
    """One `reference_benchmarks.json` entry, as `scripts/reference_results.py` scores it."""
    dataset = load_dataset(benchmark, root=root / benchmark, split=split)
    metric_names = tuple(type(m).__name__ for m in reference_results.metrics_for(dataset))
    key = reference_results.case_key(benchmark, split)
    oracle = {"combined": _load_fixture("reference_benchmarks")[key]}
    with tempfile.TemporaryDirectory() as tmp:
        reference_results.write_predictions(dataset, Path(tmp))
        return _write_real_case(
            "reference_benchmarks",
            f"{benchmark}_{split}",
            dataset,
            Path(tmp),
            metric_names,
            oracle,
            out,
        )


def _write_real_case(
    fixture: str,
    case: str,
    dataset: MOTDataset,
    pred_dir: Path,
    metric_names: tuple[str, ...],
    oracle: dict[str, Json],
    out: Path,
) -> str:
    directory = out / fixture / case
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    def writer(view: int, index: int, key: str, data: SequenceData) -> Json:
        return binary_sequence(directory, f"v{view}_s{index:03d}", key, data)

    capture = Capture(metric_names, writer)
    capture.from_dataset(dataset, pred_dir)
    body = build_case(
        fixture,
        case,
        "per_class",
        capture,
        oracle,
        raw_per_sequence=False,
        protocol=protocol_json(dataset),
    )
    (directory / "case.json").write_text(json.dumps(body, separators=(",", ":")) + "\n")
    return f"{fixture}/{case}: {body['oracle_fields_checked']} oracle fields equal"


def _run_real(job: tuple[str, str, str, Path, Path]) -> str:
    kind, a, b, root, out = job
    if kind == "real_data":
        return _real_data_case(a, root, out)
    return _reference_case(a, b, root, out)


def write_real(out: Path, jobs: int) -> None:
    root = default_data_root()
    work: list[tuple[str, str, str, Path, Path]] = [
        ("real_data", case, "", root, out) for case in REAL_DATA_CASES
    ]
    work += [("reference", b, s, root, out) for b, s in reference_results.REFERENCE_CASES]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for line in pool.map(_run_real, work):
            print(line, flush=True)
    print(f"wrote {len(work)} real-data cases to {out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--synthetic-out", type=Path, default=SYNTHETIC_OUT)
    parser.add_argument("--real-data", action="store_true", help="also dump the real-data cases")
    parser.add_argument("--real-out", type=Path, default=REAL_OUT)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes for real data")
    args = parser.parse_args(argv)
    write_synthetic(args.synthetic_out)
    if args.real_data:
        write_real(args.real_out, args.jobs)


if __name__ == "__main__":
    main()
