"""Regenerate the frozen TrackEval-parity JSON fixtures under tests/fixtures/."""

import argparse
import importlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.testdata.reference import require_reference

require_reference()

from moteval import HOTA, TrackMAP, evaluate
from moteval.benchmarks import default_data_root
from tests.scenarios import (
    BOX_METRICS,
    BOX_SCENARIOS,
    COMBINE_CLASSES_SCENARIO,
    MOTS_MULTI_CLASS_SCENARIO,
    MULTI_CLASS_BOX_PROTOCOL,
    MULTI_CLASS_MOTS_PROTOCOL,
    TRACKMAP_SCENARIOS,
    GtTracks,
    PredTracks,
    Scenario,
    box_rows_to_trackmap_spec,
    build_box_dataset,
    build_mots_scenarios,
    class_views,
    mots_track_classes,
    predictions_dir,
    prepare_dancetrack_val,
    prepare_mots20_sequence,
    prepare_sportsmot_val,
    sequence_classes,
    write_mot_scenario,
    write_mots_scenario,
)

TRACKEVAL_REPO = "https://github.com/JonathonLuiten/TrackEval"
TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
_NUMPY2_ALIAS_RE = re.compile(r"\bnp\.(float|int|bool)\b")


def ensure_trackeval_clone(cache_dir: Path) -> Path:
    if not (cache_dir / ".git").is_dir():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", TRACKEVAL_REPO, str(cache_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(cache_dir), "checkout", "--quiet", TRACKEVAL_COMMIT], check=True
    )
    apply_numpy2_patches(cache_dir / "trackeval")
    return cache_dir


def apply_numpy2_patches(package_dir: Path) -> None:
    for path in sorted(package_dir.rglob("*.py")):
        source = path.read_text()
        patched = _NUMPY2_ALIAS_RE.sub(lambda m: m.group(1), source)
        if patched != source:
            path.write_text(patched)


def import_trackeval(root: Path) -> SimpleNamespace:
    if (root / "trackeval").is_dir():
        package = "trackeval"
    elif (root / "_trackeval").is_dir():
        package = "_trackeval"
    else:
        raise SystemExit(f"no trackeval/ or _trackeval/ package under {root}")
    sys.path.insert(0, str(root))
    return SimpleNamespace(
        datasets=importlib.import_module(f"{package}.datasets"),
        metrics=importlib.import_module(f"{package}.metrics"),
        Evaluator=importlib.import_module(f"{package}.eval").Evaluator,
    )


_EVAL_CONFIG = {
    "USE_PARALLEL": False,
    "PRINT_RESULTS": False,
    "PRINT_CONFIG": False,
    "TIME_PROGRESS": False,
    "OUTPUT_SUMMARY": False,
    "OUTPUT_DETAILED": False,
    "PLOT_CURVES": False,
}


def run_mot_challenge(
    oracle: SimpleNamespace,
    gt_folder: Path,
    trackers_folder: Path,
    seq_lengths: dict[str, int],
    *,
    tracker: str = "oracle",
    benchmark: str = "MOT17",
    split: str = "train",
    do_preproc: bool = True,
    metrics: tuple[str, ...] = BOX_METRICS,
) -> dict[str, dict]:
    dataset_config = {
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(trackers_folder),
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": split,
        "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": do_preproc,
        "PRINT_CONFIG": False,
        "TRACKERS_TO_EVAL": [tracker],
        "SEQ_INFO": dict(seq_lengths),
    }
    metrics_list = [
        getattr(oracle.metrics, name)({"PRINT_CONFIG": False})
        for name in metrics
        if name != "Count"
    ]
    evaluator = oracle.Evaluator(_EVAL_CONFIG)
    dataset = oracle.datasets.MotChallenge2DBox(dataset_config)
    output_res, _ = evaluator.evaluate([dataset], metrics_list)
    combined = output_res["MotChallenge2DBox"][tracker]["COMBINED_SEQ"]["pedestrian"]
    return {name: combined[name] for name in metrics}


def run_mots_challenge(
    oracle: SimpleNamespace,
    gt_folder: Path,
    trackers_folder: Path,
    seq_lengths: dict[str, int],
    *,
    tracker: str = "oracle",
    metrics: tuple[str, ...] = ("HOTA", "CLEAR", "Identity", "Count", "JAndF"),
) -> dict[str, dict]:
    dataset_config = {
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(trackers_folder),
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
        "TRACKERS_TO_EVAL": [tracker],
        "SEQ_INFO": dict(seq_lengths),
    }
    metrics_list = [
        getattr(oracle.metrics, name)({"PRINT_CONFIG": False})
        for name in metrics
        if name != "Count"
    ]
    evaluator = oracle.Evaluator(_EVAL_CONFIG)
    dataset = oracle.datasets.MOTSChallenge(dataset_config)
    output_res, _ = evaluator.evaluate([dataset], metrics_list)
    combined = output_res["MOTSChallenge"][tracker]["COMBINED_SEQ"]["pedestrian"]
    return {name: combined[name] for name in metrics}


def _oracle_trackmap_data(gt_tracks: GtTracks, pred_tracks: PredTracks) -> dict:
    # TrackEval's TrackMAP matcher reads a stored id of 0 as unmatched (`gt_m[...] > 0`).
    gt_tracks = {tid + 1: frames for tid, frames in gt_tracks.items()}
    pred_tracks = {tid + 1: frames for tid, frames in pred_tracks.items()}
    gt_ids_sorted = sorted(gt_tracks)
    pred_ids_sorted = sorted(pred_tracks)
    dt_scores = [
        float(np.mean([c for _, c in pred_tracks[tid].values()])) for tid in pred_ids_sorted
    ]
    data = {
        "gt_track_ids": gt_ids_sorted,
        "dt_track_ids": pred_ids_sorted,
        "gt_tracks": [
            {t: np.array(box) for t, box in gt_tracks[tid].items()} for tid in gt_ids_sorted
        ],
        "dt_tracks": [
            {t: np.array(box) for t, (box, _) in pred_tracks[tid].items()}
            for tid in pred_ids_sorted
        ],
        "gt_track_areas": [
            float(np.mean([b[2] * b[3] for b in gt_tracks[tid].values()])) for tid in gt_ids_sorted
        ],
        "gt_track_lengths": [len(gt_tracks[tid]) for tid in gt_ids_sorted],
        "dt_track_areas": [
            float(np.mean([b[2] * b[3] for b, _ in pred_tracks[tid].values()]))
            for tid in pred_ids_sorted
        ],
        "dt_track_lengths": [len(pred_tracks[tid]) for tid in pred_ids_sorted],
        "dt_track_scores": np.array(dt_scores),
        "iou_type": "bbox",
        "boxformat": "xywh",
    }
    if data["dt_tracks"]:
        # TrackEval's loaders stable-sort dt tracks by descending score before eval_sequence.
        idx = np.argsort([-s for s in dt_scores], kind="mergesort")
        data["dt_track_scores"] = data["dt_track_scores"][idx]
        data["dt_tracks"] = [data["dt_tracks"][i] for i in idx]
        data["dt_track_ids"] = [data["dt_track_ids"][i] for i in idx]
        data["dt_track_areas"] = [data["dt_track_areas"][i] for i in idx]
        data["dt_track_lengths"] = [data["dt_track_lengths"][i] for i in idx]
    return data


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_fixture(path: Path, fixture: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {path}")


def gen_synthetic_box(oracle: SimpleNamespace) -> dict:
    fixture = {}
    for scenario in BOX_SCENARIOS:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seq_lengths = write_mot_scenario(tmp, scenario)
            result = run_mot_challenge(
                oracle, tmp / "gt", tmp / "trackers", seq_lengths, do_preproc=False
            )
        fixture[scenario.name] = to_jsonable(result)
    fixture["combine_classes_class_averaged"] = {"HOTA": to_jsonable(freeze_hota_class_avg(oracle))}
    fixture |= freeze_box_class_combinations(oracle)
    return fixture


def freeze_hota_class_avg(oracle: SimpleNamespace) -> dict:
    scenario = next(s for s in BOX_SCENARIOS if s.name == COMBINE_CLASSES_SCENARIO)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        write_mot_scenario(tmp, scenario)
        result = evaluate(build_box_dataset(tmp, scenario), predictions_dir(tmp), [HOTA()])
    all_res = {
        "class_a": result.per_sequence["CLASSA01"]["HOTA"],
        "class_b": result.per_sequence["CLASSB01"]["HOTA"],
    }
    return oracle.metrics.HOTA().combine_classes_class_averaged(all_res)


def _oracle_trackmap_combined(oracle: SimpleNamespace, scenario: Scenario) -> dict:
    per_seq = {
        name: oracle.metrics.TrackMAP().eval_sequence(
            _oracle_trackmap_data(*box_rows_to_trackmap_spec(gt_rows, pred_rows))
        )
        for name, _, gt_rows, pred_rows in scenario.sequences
    }
    return oracle.metrics.TrackMAP().combine_sequences(per_seq)


def _oracle_class_combinations(
    oracle: SimpleNamespace, per_class: list[dict[str, dict]], det_averaged_metrics: tuple
) -> dict:
    class_averaged = {}
    det_averaged = {}
    for name in per_class[0]:
        cls_res = {str(cls): res[name] for cls, res in enumerate(per_class, start=1)}
        metric = getattr(oracle.metrics, name)()
        class_averaged[name] = to_jsonable(metric.combine_classes_class_averaged(cls_res))
        if name in det_averaged_metrics:
            det_averaged[name] = to_jsonable(metric.combine_classes_det_averaged(cls_res))
    return {"multi_class_class_averaged": class_averaged, "multi_class_det_averaged": det_averaged}


def freeze_box_class_combinations(oracle: SimpleNamespace) -> dict:
    scenario = next(s for s in BOX_SCENARIOS if s.name == COMBINE_CLASSES_SCENARIO)
    per_class = []
    classes = MULTI_CLASS_BOX_PROTOCOL.eval_classes
    for view in class_views(scenario, sequence_classes(scenario), classes):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seq_lengths = write_mot_scenario(tmp, view)
            result = run_mot_challenge(
                oracle, tmp / "gt", tmp / "trackers", seq_lengths, do_preproc=False
            )
        result["TrackMAP"] = _oracle_trackmap_combined(oracle, view)
        per_class.append(result)
    # TrackEval's TrackMAP combine_classes_det_averaged copies its class-averaged combiner.
    return _oracle_class_combinations(oracle, per_class, det_averaged_metrics=BOX_METRICS)


def freeze_mots_class_combinations(oracle: SimpleNamespace) -> dict:
    scenario = next(s for s in build_mots_scenarios() if s.name == MOTS_MULTI_CLASS_SCENARIO)
    per_class = []
    classes = MULTI_CLASS_MOTS_PROTOCOL.eval_classes
    for view in class_views(scenario, mots_track_classes, classes):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seq_lengths = write_mots_scenario(tmp, view)
            per_class.append(run_mots_challenge(oracle, tmp / "gt", tmp / "trackers", seq_lengths))
    return _oracle_class_combinations(oracle, per_class, det_averaged_metrics=tuple(per_class[0]))


def gen_synthetic_mots(oracle: SimpleNamespace) -> dict:
    fixture = {}
    for scenario in build_mots_scenarios():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seq_lengths = write_mots_scenario(tmp, scenario)
            result = run_mots_challenge(oracle, tmp / "gt", tmp / "trackers", seq_lengths)
        fixture[scenario.name] = to_jsonable(result)
    fixture |= freeze_mots_class_combinations(oracle)
    return fixture


def gen_synthetic_trackmap(oracle: SimpleNamespace) -> dict:
    trackmap_fields = TrackMAP().fields
    fixture = {}
    for name, sequences in TRACKMAP_SCENARIOS.items():
        per_seq = {
            seq_name: oracle.metrics.TrackMAP().eval_sequence(
                _oracle_trackmap_data(gt_tracks, pred_tracks)
            )
            for seq_name, (_, gt_tracks, pred_tracks) in sequences.items()
        }
        combined = oracle.metrics.TrackMAP().combine_sequences(per_seq)
        fixture[name] = {"TrackMAP": {f: to_jsonable(combined[f]) for f in trackmap_fields}}
    return fixture


def gen_real_data(oracle: SimpleNamespace) -> dict:
    for benchmark in ("dancetrack", "sportsmot", "mots20"):
        root = default_data_root() / benchmark
        if not root.is_dir() or not any(root.iterdir()):
            raise SystemExit(
                f"{benchmark} not found under {root} — fetch it with "
                f"`scripts/download_benchmarks.py download {benchmark}`"
            )
    fixture = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, gt_root, seq_lengths = prepare_dancetrack_val(tmp)
        fixture["dancetrack_val"] = to_jsonable(
            run_mot_challenge(oracle, gt_root, tmp / "trackers", seq_lengths)
        )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, gt_root, seq_lengths = prepare_sportsmot_val(tmp)
        fixture["sportsmot_val"] = to_jsonable(
            run_mot_challenge(oracle, gt_root, tmp / "trackers", seq_lengths)
        )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, gt_root, seq_lengths = prepare_mots20_sequence(tmp)
        fixture["mots20"] = to_jsonable(
            run_mots_challenge(oracle, gt_root, tmp / "trackers", seq_lengths)
        )
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--trackeval-dir",
        type=Path,
        help="existing TrackEval checkout (or the old vendored tests/oracle dir); "
        "default: clone the pinned commit into --cache-dir",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "moteval" / "trackeval",
        help="clone destination when --trackeval-dir is not given",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "tests" / "fixtures",
        help="where the fixture JSON files are written",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="also regenerate real_data.json from $MOTEVAL_DATA_ROOT, else data/benchmarks/",
    )
    args = parser.parse_args()

    trackeval_dir = args.trackeval_dir or ensure_trackeval_clone(args.cache_dir)
    oracle = import_trackeval(trackeval_dir.resolve())

    write_fixture(args.output_dir / "synthetic_box.json", gen_synthetic_box(oracle))
    write_fixture(args.output_dir / "synthetic_mots.json", gen_synthetic_mots(oracle))
    write_fixture(args.output_dir / "synthetic_trackmap.json", gen_synthetic_trackmap(oracle))
    if args.real_data:
        write_fixture(args.output_dir / "real_data.json", gen_real_data(oracle))


if __name__ == "__main__":
    main()
