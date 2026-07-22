"""Regenerate the frozen TrackEval-parity JSON fixtures under tests/fixtures/.

The oracle is official TrackEval pinned at commit
12c8791b303e0a0b50f753af204249e622d0281a, cloned on demand into a cache dir and
patched in place for numpy>=2 (``np.float``->``float``, ``np.int``->``int``,
``np.bool``->``bool``, word-boundary matching only, so ``np.float64``/``np.int32``/
``np.bool_`` etc. are never touched — the exact patch set documented in the old
tests/oracle/VENDORED.md).

Run from the repo root:

    uv run --with opencv-python-headless --with scikit-image \
        python scripts/regen_parity_fixtures.py

(cv2/scikit-image are needed only by the oracle's JAndF metric on the MOTS path;
moteval itself needs neither.)

Modes:
  default            regenerate the synthetic fixtures: synthetic_box.json,
                     synthetic_mots.json, synthetic_trackmap.json
  --real-data        additionally regenerate real_data.json from data/benchmarks/
                     (DanceTrack val, SportsMOT val, one MOTS20 sequence) using
                     tests/perturb.py seeded predictions
  --trackeval-dir    use an existing TrackEval checkout (package dir ``trackeval``)
                     or the old vendored dir (``tests/oracle``, package
                     ``_trackeval``) instead of cloning

Fixture JSON format: ``{scenario_name: {metric: {field: scalar-or-list}}}``. Floats
go through plain ``json`` (Python round-trips doubles exactly); arrays become lists.
synthetic_box.json additionally holds a ``combine_classes_class_averaged`` entry:
moteval's per-sequence HOTA results for the combine_classes scenario fed to the
oracle HOTA ``combine_classes_class_averaged`` combiner. ``det_averaged`` is NOT
frozen — upstream's is a copy-paste bug and moteval's corrected version is the sole
intentional numeric divergence (ADR-0001).
"""

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
    sys.path.insert(0, str(_REPO_ROOT))

from moteval import HOTA, TrackMAP, evaluate  # noqa: E402
from tests.scenarios import (  # noqa: E402
    BOX_METRICS,
    BOX_SCENARIOS,
    COMBINE_CLASSES_SCENARIO,
    DATA_ROOT,
    TRACKMAP_SCENARIOS,
    GtTracks,
    PredTracks,
    build_box_dataset,
    build_mots_scenarios,
    predictions_dir,
    prepare_dancetrack_val,
    prepare_mots20_sequence,
    prepare_sportsmot_val,
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
    """Replace removed numpy aliases with the exact builtins they aliased.

    Word-boundary matching only: ``np.float64``, ``np.int32``, ``np.bool_``,
    ``np.floating`` etc. never match. Idempotent, so safe on every run.
    """
    for path in sorted(package_dir.rglob("*.py")):
        source = path.read_text()
        patched = _NUMPY2_ALIAS_RE.sub(lambda m: m.group(1), source)
        if patched != source:
            path.write_text(patched)


def import_trackeval(root: Path) -> SimpleNamespace:
    """Import the oracle package from ``root`` (auto-detects the package name)."""
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
    do_preproc: bool = True,  # TrackEval's own default; the real-data gate keeps it
    metrics: tuple[str, ...] = BOX_METRICS,
) -> dict[str, dict]:
    """Evaluate a MOTChallenge-format gt/trackers pair through the oracle.

    Ported from the old tests/oracle/runner.py, parameterized on the imported
    package. Returns the ``COMBINED_SEQ`` pedestrian results per metric.
    """
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
    """Evaluate a MOTS-format gt/trackers pair through the oracle (JAndF needs cv2)."""
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
    """Build the upstream-shaped TrackMAP ``data`` dict from a scenario spec.

    Ids are offset by +1: upstream's greedy matcher tracks "already matched" via
    ``gt_m[thr, gt] > 0`` against the literal matched id, which misbehaves for a
    legitimate id of 0 (real TrackEval datasets never emit id 0; moteval fixes it).
    Sorted order is preserved, and id values otherwise never affect the numbers,
    so this compares the semantics upstream clearly intends.
    """
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
        # Upstream dataset loaders presort dt tracks by descending score (mergesort,
        # so ties keep original order) before eval_sequence ever runs.
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
    return fixture


def freeze_hota_class_avg(oracle: SimpleNamespace) -> dict:
    """Oracle HOTA class-averaged combiner over moteval's own per-sequence results.

    The MOTChallenge runner only ever evaluates one class, so the combiner can't be
    exercised through it: moteval's deterministic per-sequence HOTA dicts for the
    combine_classes scenario (CLASSA01/CLASSB01) stand in as two classes.
    """
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


def gen_synthetic_mots(oracle: SimpleNamespace) -> dict:
    fixture = {}
    for scenario in build_mots_scenarios():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seq_lengths = write_mots_scenario(tmp, scenario)
            result = run_mots_challenge(oracle, tmp / "gt", tmp / "trackers", seq_lengths)
        fixture[scenario.name] = to_jsonable(result)
    return fixture


def gen_synthetic_trackmap(oracle: SimpleNamespace) -> dict:
    trackmap_fields = TrackMAP().fields  # drop oracle-internal precision/recall work arrays
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
        root = DATA_ROOT / benchmark
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
        help="also regenerate real_data.json from data/benchmarks/",
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
