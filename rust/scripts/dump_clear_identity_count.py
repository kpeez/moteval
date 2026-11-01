"""Dump seeded random `SequenceData` cases scored by Python CLEAR, Identity, and Count.

The parity fixtures cover the common paths. These cases add what they do not: a gt or
pred id repeated within one frame (numpy fancy-index ``+=`` counts it once; ``bincount``
counts every occurrence), similarities on both sides of CLEAR's ``0.5 - eps`` guard and
Identity's unguarded ``0.5``, ties, empty frames, the two early returns, and class
combiners over one to four classes. The last cases combine 8 to 12 classes of dense
sequences with 8 or more matches per frame, where numpy's pairwise ``np.mean`` (class
averaging) and the plain sum of ``MOTP_sum`` differ from the other order.

Each case is one file in the scenario case format (``rust/testdata/README.md``) with an
empty ``oracle``: the expected numbers are Python moteval's own (``moteval`` section),
per sequence (raw), per view combined, class-averaged, and det-averaged.

    artifacts/rust-port/ref-venv/bin/python rust/scripts/dump_clear_identity_count.py
"""

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
# After site-packages: `moteval` is the pure-Python reference (rust/scripts/reference.py).
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from rust.scripts.reference import require_reference

require_reference()

from moteval import SequenceData
from moteval.data.model import BoxGeometry
from rust.scripts import dump_scenarios as ds

OUT = Path(__file__).resolve().parents[1] / "testdata" / "fuzz" / "clear_identity_count"
METRICS = ("CLEAR", "Identity", "Count")
NUM_CASES = 48
NUM_MANY_CLASS_CASES = 6
SEED = 20261031
EPS = np.finfo(float).eps
# Values that sit on a threshold: CLEAR keeps `sim >= 0.5 - eps`, Identity keeps
# `sim >= 0.5`, and the tracked-ratio cuts are 0.8 and 0.2.
SPECIAL = np.array(
    [
        0.0,
        -0.0,
        0.5,
        0.5 - EPS,
        np.nextafter(0.5 - EPS, 0.0),
        np.nextafter(0.5 - EPS, 1.0),
        np.nextafter(0.5, 0.0),
        np.nextafter(0.5, 1.0),
        0.25,
        0.75,
        1.0,
        0.8,
        0.2,
    ]
)


def draw_similarity(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    kind = rng.integers(0, 4, size=shape)
    special = SPECIAL[rng.integers(0, len(SPECIAL), size=shape)]
    uniform = rng.random(shape)
    return np.where(kind == 0, special, np.where(kind == 1, uniform * 0.5, uniform))


def draw_ids(rng: np.random.Generator, num_ids: int, repeat_rate: float) -> np.ndarray:
    if num_ids == 0:
        return np.zeros(0, dtype=np.int64)
    ids = rng.choice(num_ids, size=int(rng.integers(0, num_ids + 1)), replace=False)
    if len(ids) and rng.random() < repeat_rate:
        ids = np.append(ids, rng.choice(ids, size=int(rng.integers(1, 3))))
        rng.shuffle(ids)
    return ids.astype(np.int64)


def densify(frames: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Map the ids that occur to ``0..n-1`` in sorted order, as `convert` does."""
    used = sorted({int(i) for frame in frames for i in frame})
    mapping = {raw: dense for dense, raw in enumerate(used)}
    return [np.array([mapping[int(i)] for i in f], dtype=np.int64) for f in frames], len(used)


def random_sequence(rng: np.random.Generator, name: str, large: bool) -> SequenceData:
    num_timesteps = int(rng.integers(1, 41 if large else 13))
    raw_gt = int(rng.integers(0, 16 if large else 6))
    raw_pred = int(rng.integers(0, 16 if large else 6))
    mode = int(rng.integers(0, 10))  # 0: no predictions, 1: no ground truth
    repeat_rate = 0.0 if rng.random() < 0.5 else 0.15
    base = draw_similarity(rng, (max(raw_gt, 1), max(raw_pred, 1)))
    gt_frames, pred_frames, sims = [], [], []
    for _ in range(num_timesteps):
        gt = draw_ids(rng, 0 if mode == 1 else raw_gt, repeat_rate)
        pred = draw_ids(rng, 0 if mode == 0 else raw_pred, repeat_rate)
        sim = base[np.ix_(gt, pred)].copy()
        jitter = rng.random(sim.shape) < 0.3
        sim[jitter] = draw_similarity(rng, sim.shape)[jitter]
        gt_frames.append(gt)
        pred_frames.append(pred)
        sims.append(sim)
    gt_ids, num_gt_ids = densify(gt_frames)
    pred_ids, num_pred_ids = densify(pred_frames)
    return SequenceData(
        name=name,
        num_timesteps=num_timesteps,
        num_gt_ids=num_gt_ids,
        num_pred_ids=num_pred_ids,
        num_gt_dets=sum(len(f) for f in gt_ids),
        num_pred_dets=sum(len(f) for f in pred_ids),
        gt_ids=tuple(gt_ids),
        pred_ids=tuple(pred_ids),
        pred_confidences=tuple(np.zeros(len(f)) for f in pred_ids),
        geometry=BoxGeometry(
            gt=tuple(np.zeros((len(f), 4)) for f in gt_ids),
            pred=tuple(np.zeros((len(f), 4)) for f in pred_ids),
        ),
        similarity=tuple(sims),
    )


def dense_sequence(rng: np.random.Generator, name: str) -> SequenceData:
    """Every id in every frame, each GT paired at or above 0.5 with one prediction.

    Each frame has 8 or more matches, where numpy's pairwise sum and a plain sum of the
    matched similarities (CLEAR's ``MOTP_sum``) can differ.
    """
    num_timesteps = int(rng.integers(2, 4))
    n = int(rng.integers(8, 11))
    gt_frames, pred_frames, sims = [], [], []
    for _ in range(num_timesteps):
        gt = rng.permutation(n).astype(np.int64)
        pred = rng.permutation(n).astype(np.int64)
        sim = rng.random((n, n)) * 0.5
        sim[np.arange(n), rng.permutation(n)] = 0.5 + 0.5 * rng.random(n)
        gt_frames.append(gt)
        pred_frames.append(pred)
        sims.append(sim)
    return SequenceData(
        name=name,
        num_timesteps=num_timesteps,
        num_gt_ids=n,
        num_pred_ids=n,
        num_gt_dets=n * num_timesteps,
        num_pred_dets=n * num_timesteps,
        gt_ids=tuple(gt_frames),
        pred_ids=tuple(pred_frames),
        pred_confidences=tuple(np.zeros(n) for _ in pred_frames),
        geometry=BoxGeometry(
            gt=tuple(np.zeros((n, 4)) for _ in gt_frames),
            pred=tuple(np.zeros((n, 4)) for _ in pred_frames),
        ),
        similarity=tuple(sims),
    )


def build_case(index: int, rng: np.random.Generator) -> ds.Json:
    capture = ds.Capture(METRICS, ds._inline_writer)
    large = index % 8 == 7
    # The cases after NUM_CASES combine 8 to 12 classes of dense sequences: below 8
    # values numpy's pairwise mean equals a plain one, so fewer classes cannot tell them
    # apart. They come last, so the earlier cases' draws do not change.
    many_classes = index >= NUM_CASES
    num_classes = int(rng.integers(8, 13)) if many_classes else int(rng.integers(1, 5))
    for cls in range(1, num_classes + 1):
        view = ds.View(cls)
        capture.views.append(view)
        for s in range(1 if many_classes else int(rng.integers(1, 5))):
            key = f"seq{s}"
            sequence = (
                dense_sequence(rng, key) if many_classes else random_sequence(rng, key, large)
            )
            capture.add(view, key, sequence)
    results = ds.score(capture, "per_class")
    return {
        "format": ds.FORMAT_VERSION,
        "fixture": "fuzz_clear_identity_count",
        "case": f"case_{index:03d}",
        "procedure": "per_class",
        "protocol": None,
        "metrics": list(METRICS),
        "views": [{"class": v.cls, "sequences": v.records} for v in capture.views],
        "oracle": {},
        "moteval": {
            "views": [
                {
                    "class": v.cls,
                    "per_sequence": [
                        [key, ds.encode_metric_scores(scores)] for key, scores in v.per_sequence
                    ],
                    "combined": ds.encode_metric_scores(comb),
                }
                for v, comb in zip(capture.views, results["combined"], strict=True)
            ],
            "class_averaged": ds.encode_metric_scores(results["class_averaged"]),
            "det_averaged": ds.encode_metric_scores(results["det_averaged"]),
        },
        "oracle_fields_checked": 0,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("case_*.json"):
        old.unlink()
    total = NUM_CASES + NUM_MANY_CLASS_CASES
    for index in range(total):
        case = build_case(index, rng)
        path = OUT / f"{case['case']}.json"
        path.write_text(json.dumps(case, separators=(",", ":")) + "\n")
    print(f"wrote {total} cases to {OUT}")


if __name__ == "__main__":
    main()
