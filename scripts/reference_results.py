"""Score the nine local benchmarks with seeded perturbed predictions and write JSON.

For every (benchmark, split) in `REFERENCE_CASES` the script loads the ground truth
from the data root (``--root``, else ``MOTEVAL_DATA_ROOT``, else ``data/benchmarks``),
writes ``tests/perturb.py`` predictions (sequence ``i`` gets seed
``tests.scenarios.REAL_DATA_SEED + i``), and runs every applicable metric: HOTA, CLEAR,
Identity, Count and TrackMAP on boxes; HOTA, CLEAR, Identity, Count and J&F on masks.

    uv run python scripts/reference_results.py --out-dir artifacts/rust-port/baseline/after
    uv run python scripts/reference_results.py --out-dir <dir> --reference-fixture \
        tests/fixtures/reference_benchmarks.json

``--out-dir`` receives one ``<benchmark>_<split>.json`` per case in the
`moteval.results.to_json_dict` shape (per-sequence and combined, floats exact, NaN as
Python's ``json`` writes it). ``--reference-fixture`` also writes one JSON file of the
combined results only, keyed ``<benchmark>/<split>``.
"""

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moteval import CLEAR, HOTA, Count, Identity, JAndF, Metric, MOTDataset, TrackMAP, evaluate
from moteval.benchmarks import default_data_root, load_dataset
from moteval.data.model import MaskGtSequence
from moteval.formats import write_mot, write_mots
from moteval.results import JsonResult, to_json_dict
from tests.perturb import perturb_box_tracks, perturb_mask_tracks
from tests.scenarios import REAL_DATA_SEED

REFERENCE_CASES: tuple[tuple[str, str], ...] = (
    ("animaltrack", "all"),
    ("animaltrack", "train"),
    ("animaltrack", "test"),
    ("bft", "train"),
    ("bft", "val"),
    ("bft", "test"),
    ("chimpact", "train"),
    ("chimpact", "val"),
    ("chimpact", "test"),
    ("dancetrack", "train"),
    ("dancetrack", "val"),
    ("gmot40", "test"),
    ("gmot40", "animal"),
    ("mots20", "train"),
    ("panaf500", "train"),
    ("panaf500", "validation"),
    ("panaf500", "test"),
    ("sportsmot", "train"),
    ("sportsmot", "val"),
    ("uavdt", "all"),
)


def case_key(benchmark: str, split: str) -> str:
    return f"{benchmark}/{split}"


def write_predictions(dataset: MOTDataset, pred_dir: Path) -> None:
    """Write seeded perturbed predictions: sequence ``i`` gets seed ``REAL_DATA_SEED + i``."""
    convention = dataset.protocol.frame_convention
    for i, seq in enumerate(dataset.sequences):
        seed = REAL_DATA_SEED + i
        if isinstance(seq, MaskGtSequence):
            masks = perturb_mask_tracks(seq.tracks, seq.num_timesteps, convention, seed=seed)
            write_mots(pred_dir / f"{seq.name}.txt", masks)
        else:
            boxes = perturb_box_tracks(seq.tracks, seq.num_timesteps, convention, seed=seed)
            write_mot(pred_dir / f"{seq.name}.txt", boxes)


def metrics_for(dataset: MOTDataset) -> list[Metric]:
    """Every applicable metric: TrackMAP on boxes, J&F on masks."""
    is_masks = isinstance(dataset.sequences[0], MaskGtSequence)
    return [HOTA(), CLEAR(), Identity(), Count(), JAndF() if is_masks else TrackMAP()]


def evaluate_case(benchmark: str, split: str, root: Path) -> JsonResult:
    """Load one benchmark split, score seeded perturbed predictions, return the JSON dict."""
    dataset = load_dataset(benchmark, root=root / benchmark, split=split)
    with tempfile.TemporaryDirectory() as tmp:
        write_predictions(dataset, Path(tmp))
        result = evaluate(dataset, tmp, metrics_for(dataset))
    return to_json_dict(result, dataset=benchmark, split=split)


def _run_case(args: tuple[str, str, Path, Path]) -> str:
    benchmark, split, root, out_dir = args
    scored = evaluate_case(benchmark, split, root)
    (out_dir / f"{benchmark}_{split}.json").write_text(json.dumps(scored))
    return case_key(benchmark, split)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--reference-fixture", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.root if args.root is not None else default_data_root()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = [(b, s, root, args.out_dir) for b, s in REFERENCE_CASES]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for key in pool.map(_run_case, work):
            print(f"done {key}", flush=True)

    if args.reference_fixture is not None:
        combined = {
            case_key(b, s): json.loads((args.out_dir / f"{b}_{s}.json").read_text())["combined"]
            for b, s in REFERENCE_CASES
        }
        args.reference_fixture.write_text(json.dumps(combined, indent=1) + "\n")


if __name__ == "__main__":
    main()
