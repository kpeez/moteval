"""Dump evaluation results with the exact table, JSON, and CSV bytes Python writes.

Writes ``rust/testdata/exports.json``, which ``rust/moteval/tests/exports.rs`` checks byte
for byte, and the messages of `evaluate`'s errors before scoring. Run with the reference
interpreter from the repository root:

    artifacts/rust-port/ref-venv/bin/python rust/scripts/dump_results.py

Each case holds an `EvaluationResult` (scores as ``{"shape", "bits"}``, in dict order),
plus what ``moteval.cli`` prints and writes for it: ``cli._table`` (or the exception it
raises), ``json.dumps(to_json_dict(...))``, and the bytes ``cli._write_csv`` writes. Cases
come from real `evaluate` runs on the test suite's toy and two-class datasets and from
hand-built results with adversarial values and names.
"""

import json
import math
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
# After site-packages: `moteval` is the pure-Python reference (rust/scripts/reference.py).
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from rust.scripts.reference import require_reference

require_reference()

from moteval import (
    CLEAR,
    HOTA,
    Count,
    FrameConvention,
    GtSequence,
    Identity,
    JAndF,
    MaskGtSequence,
    MOTDataset,
    Protocol,
    TrackMAP,
    evaluate,
)
from moteval.cli import _table, _write_csv  # pyrefly: ignore[missing-module-attribute]
from moteval.results import ClassResult, EvaluationResult, to_json_dict
from rust.scripts.testgen import SplitMix64, f64_bits
from tests.conftest import (
    load_toy,
    load_two_class,
    write_flawed_predictions,
    write_perfect_predictions,
    write_two_class_predictions,
)

OUT = Path(__file__).resolve().parents[1] / "testdata" / "exports.json"
METRICS = {"HOTA": HOTA, "CLEAR": CLEAR, "Identity": Identity, "Count": Count}
METRICS |= {"TrackMAP": TrackMAP, "JAndF": JAndF}


def enc_value(value) -> dict:
    if not isinstance(value, (float, np.floating, np.ndarray)):
        raise TypeError(f"score value of type {type(value).__name__}: Rust has no such Value")
    arr = np.asarray(value, dtype=np.float64)
    return {"shape": list(arr.shape), "bits": [f64_bits(float(x)) for x in arr.ravel()]}


def enc_metric_scores(scores: dict) -> list:
    return [
        [metric, [[field, enc_value(value)] for field, value in fields.items()]]
        for metric, fields in scores.items()
    ]


def enc_per_sequence(per_sequence: dict) -> list:
    return [[seq, enc_metric_scores(scores)] for seq, scores in per_sequence.items()]


def enc_result(result: EvaluationResult) -> dict:
    return {
        "per_sequence": enc_per_sequence(result.per_sequence),
        "combined": enc_metric_scores(result.combined),
        "per_class": [
            [
                cls_id,
                {
                    "per_sequence": enc_per_sequence(class_result.per_sequence),
                    "combined": enc_metric_scores(class_result.combined),
                },
            ]
            for cls_id, class_result in result.per_class.items()
        ],
        "class_averaged": enc_metric_scores(result.class_averaged),
        "det_averaged": enc_metric_scores(result.det_averaged),
    }


def case(name: str, result: EvaluationResult, dataset: str, split: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # np.mean of an empty array
        try:
            table = {"text": _table(result)}
        except (ValueError, OverflowError, KeyError) as err:
            table = {"error": type(err).__name__, "message": str(err)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            _write_csv(path, result)
            csv_text = path.read_bytes().decode()
    return {
        "name": name,
        "dataset": dataset,
        "split": split,
        "result": enc_result(result),
        "table": table,
        "json": json.dumps(to_json_dict(result, dataset=dataset, split=split)),
        "csv": csv_text,
    }


def evaluated_cases() -> list[dict]:
    all_box = ["HOTA", "CLEAR", "Identity", "Count", "TrackMAP"]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        toy = load_toy()
        perfect, flawed, two = Path(tmp, "perfect"), Path(tmp, "flawed"), Path(tmp, "two")
        write_perfect_predictions(toy, perfect)
        flawed.mkdir()
        write_flawed_predictions(toy, flawed)
        write_two_class_predictions(two)
        two_class = load_two_class()
        runs = [
            ("toy_perfect", toy, perfect, all_box),
            ("toy_flawed", toy, flawed, all_box),
            ("toy_count", toy, perfect, ["Count"]),
            ("toy_trackmap_only", toy, flawed, ["TrackMAP"]),
            ("two_class", two_class, two, all_box),
            ("two_class_hota_count", two_class, two, ["HOTA", "Count"]),
        ]
        for name, dataset, pred_dir, metrics in runs:
            result = evaluate(dataset, pred_dir, [METRICS[m]() for m in metrics])
            out.append(case(name, result, dataset.name, dataset.split))
    return out


HOTA_ARRAYS = ("HOTA", "DetA", "AssA", "LocA")


def adversarial_scores(rng: SplitMix64, extremes: list[float]) -> dict:
    """Headline metrics with arrays whose mean depends on summation order and scalars
    drawn from `extremes`."""

    def pick() -> float:
        return extremes[rng.below(len(extremes))]

    def array(n: int) -> np.ndarray:
        return np.array([rng.spread() if rng.below(3) else pick() for _ in range(n)])

    return {
        "HOTA": {name: array(19) for name in HOTA_ARRAYS} | {"HOTA(0)": pick()},
        "CLEAR": {"MOTA": pick(), "MOTP": pick(), "IDSW": float(rng.below(1000)) - 0.5},
        "Identity": {"IDF1": pick(), "IDTP": pick()},
        "Count": {"Dets": rng.spread(), "GT_Dets": float(rng.below(1 << 60)) * 3.0},
        "Extra": {
            "matrix": array(15).reshape(3, 5),
            "empty": np.array([], dtype=np.float64),
            "one": array(1),
            "long": array(137),  # numpy's pairwise sum recurses above 128
        },
    }


EXTREMES = [
    0.0, -0.0, 1.0, 0.5, 1e-7, 1e-5, 0.0001, 1e16, 1e22, 0.1 + 0.2, 2 / 3, 0.99999,
    0.999995, 1.0000049999, 5e-324, 1.7976931348623157e308, math.nan, math.inf, -math.inf,
    -2.5, 123456.5, 0.0009999995,
]  # fmt: skip
SEQ_NAMES = ["plain", "a,b", 'q"uote', "new\nline", "cr\rx", "é日本😀", "tab\tx", "x" * 30]


def finite(scores: dict) -> dict:
    """Integer headlines must convert with `int()`: keep them finite."""
    for metric, field in (("CLEAR", "IDSW"), ("Count", "Dets"), ("Count", "GT_Dets")):
        value = scores[metric][field]
        if not math.isfinite(value):
            scores[metric][field] = 7.0
    return scores


def adversarial_cases() -> list[dict]:
    rng = SplitMix64(4242)
    out = []
    single = EvaluationResult(
        per_sequence={name: finite(adversarial_scores(rng, EXTREMES)) for name in SEQ_NAMES},
        combined=finite(adversarial_scores(rng, EXTREMES)),
    )
    out.append(case("adversarial_single", single, 'd"s é\n', "sp,lit"))

    classes = {3: SEQ_NAMES[:3], -1: SEQ_NAMES[3:], 10: []}
    multi = EvaluationResult(
        per_sequence={},
        combined={},
        per_class={
            cls_id: ClassResult(
                per_sequence={name: finite(adversarial_scores(rng, EXTREMES)) for name in names},
                combined=finite(adversarial_scores(rng, EXTREMES)),
            )
            for cls_id, names in classes.items()
        },
        class_averaged=finite(adversarial_scores(rng, EXTREMES)),
        det_averaged=finite(adversarial_scores(rng, EXTREMES)),
    )
    out.append(case("adversarial_multi", multi, "multi", "val"))

    # Counts that int() truncates toward zero, huge and negative ones included.
    counts = EvaluationResult(
        per_sequence={
            f"s{i}": {"Count": {"Dets": dets, "GT_Dets": gt}}
            for i, (dets, gt) in enumerate(
                [(3.7, -2.5), (-0.0, 0.4), (1e20, 2.0**63), (-1e300, 9007199254740993.0)]
            )
        },
        combined={"Count": {"Dets": 1.5e16, "GT_Dets": -0.999}},
    )
    out.append(case("integer_truncation", counts, "counts", "val"))

    nan_count = EvaluationResult(
        per_sequence={"s": {"Count": {"Dets": math.nan, "GT_Dets": 1.0}}},
        combined={"Count": {"Dets": 1.0, "GT_Dets": 1.0}},
    )
    out.append(case("table_nan_count", nan_count, "nan", "val"))
    inf_count = EvaluationResult(
        per_sequence={"s": {"CLEAR": {"MOTA": 0.5, "MOTP": 0.5, "IDSW": math.inf}}},
        combined={"CLEAR": {"MOTA": 0.5, "MOTP": 0.5, "IDSW": 1.0}},
    )
    out.append(case("table_inf_count", inf_count, "inf", "val"))
    missing = EvaluationResult(
        per_sequence={"s": {}},
        combined={"Count": {"Dets": 1.0, "GT_Dets": 1.0}},
    )
    out.append(case("table_missing_metric", missing, "missing", "val"))

    out.append(case("empty", EvaluationResult(per_sequence={}, combined={}), "", ""))
    no_headlines = EvaluationResult(
        per_sequence={"é日本😀": {"Extra": {"x": 1.0}}},
        combined={"Extra": {"x": np.array([1.0, 2.0])}},
    )
    out.append(case("no_headline_columns", no_headlines, "x", "y"))
    return out


# ------------------------------------------------------------- errors before scoring


def build_dataset(spec: dict) -> MOTDataset:
    sequences = tuple(
        MaskGtSequence(name=s["name"], num_timesteps=1, tracks=())
        if s["kind"] == "masks"
        else GtSequence(name=s["name"], num_timesteps=1, tracks=())
        for s in spec["sequences"]
    )
    protocol = Protocol(
        name=spec["protocol"],
        frame_convention=FrameConvention(name="1-indexed", first_frame=1),
        eval_classes=tuple(spec["eval_classes"]),
    )
    return MOTDataset(name=spec["name"], split="val", sequences=sequences, protocol=protocol)


def boxes(*names: str) -> list[dict]:
    return [{"name": name, "kind": "boxes"} for name in names]


def masks(*names: str) -> list[dict]:
    return [{"name": name, "kind": "masks"} for name in names]


ERROR_CASES = [
    # (name, dataset spec, metrics, prediction files present)
    ("duplicate_metrics", {"name": "d", "protocol": "p", "eval_classes": [1],
     "sequences": boxes("s1")}, ["HOTA", "CLEAR", "HOTA", "Count", "CLEAR", "CLEAR"], []),
    ("duplicate_before_geometry", {"name": "d", "protocol": "p", "eval_classes": [],
     "sequences": boxes("s1")}, ["JAndF", "JAndF"], []),
    ("no_eval_classes", {"name": "d", "protocol": "it's", "eval_classes": [],
     "sequences": boxes("s1")}, ["HOTA"], []),
    ("repeated_class", {"name": "d", "protocol": "p", "eval_classes": [1, 2, 1],
     "sequences": boxes("s1")}, ["HOTA"], []),
    ("repeated_single_class", {"name": "d", "protocol": "p", "eval_classes": [5, 5],
     "sequences": boxes("s1")}, ["HOTA"], []),
    ("jf_on_boxes", {"name": "toy", "protocol": "p", "eval_classes": [1],
     "sequences": boxes("toy-0001", "toy-0002")}, ["HOTA", "JAndF"], []),
    ("trackmap_on_masks", {"name": "mots", "protocol": "p", "eval_classes": [1],
     "sequences": masks("m'1")}, ["TrackMAP", "JAndF"], []),
    ("geometry_metric_order", {"name": "mixed", "protocol": "p", "eval_classes": [1, 2],
     "sequences": masks("m1") + boxes("b1")}, ["JAndF", "TrackMAP"], []),
    ("geometry_before_missing", {"name": "d", "protocol": "p", "eval_classes": [1],
     "sequences": boxes("s1")}, ["TrackMAP", "JAndF"], []),
    ("missing_some", {"name": "d", "protocol": "p", "eval_classes": [1],
     "sequences": boxes("s1", "s2", "s3") + masks("m1")}, ["HOTA"], ["s2"]),
    ("missing_all", {"name": "d", "protocol": "p", "eval_classes": [1],
     "sequences": boxes("é日本", "b")}, ["Count"], []),
]  # fmt: skip


def error_cases() -> list[dict]:
    out = []
    for name, spec, metrics, files in ERROR_CASES:
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp) / "preds"
            pred_dir.mkdir()
            for stem in files:
                (pred_dir / f"{stem}.txt").write_text("")
            # The path as given; the message prints it normalized.
            given = f"{pred_dir}//./"
            try:
                evaluate(build_dataset(spec), given, [METRICS[m]() for m in metrics])
            except ValueError as err:
                message = str(err).replace(str(pred_dir), "<PRED>")
            else:
                raise AssertionError(f"{name}: evaluate did not raise")
        out.append(
            {"name": name, "dataset": spec, "metrics": metrics, "files": files,
             "message": message}
        )  # fmt: skip
    return out


def main() -> None:
    data = {
        "format": 1,
        "python": sys.version.split()[0],
        "cases": evaluated_cases() + adversarial_cases(),
        "errors": error_cases(),
    }
    # One case per line keeps the file under the repository's 500 KB hook limit.
    lines = ",\n".join(json.dumps(item, separators=(",", ":")) for item in data["cases"])
    errors = ",\n".join(json.dumps(item) for item in data["errors"])
    head = json.dumps({k: data[k] for k in ("format", "python")})[:-1]
    OUT.write_text(f'{head}, "cases": [\n{lines}\n], "errors": [\n{errors}\n]}}\n')
    print(f"wrote {OUT} ({len(data['cases'])} result cases, {len(data['errors'])} errors)")


if __name__ == "__main__":
    main()
