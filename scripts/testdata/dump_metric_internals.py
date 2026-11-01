"""Dump HOTA and TrackMAP intermediates and combiners the scenario fixtures cannot reach.

The scenario fixtures check every public HOTA and TrackMAP output, but some reduction
orders only show in their last bits, and those bits rarely change an output: HOTA's
``similarity.sum(1)`` (pairwise once a frame has 8 or more predictions) only moves the
global alignment score that feeds the assignment, the class combiners only reduce
pairwise with 8 or more classes, and TrackMAP's track areas only meet the area-range
bounds. This script checks them directly, through moteval's own code, on seeded inputs:

- ``hota_dense``: dense sequences (8 to 24 ids, most frames with 8 or more detections).
  It records every cost matrix HOTA hands to ``linear_sum_assignment`` and the scores
  of ``eval_sequence`` and ``combine_sequences``.
- ``hota_class_combiners`` and ``trackmap_class_combiners``: 8 to 20 classes of random
  per-class scores through ``combine_classes_class_averaged`` and
  ``combine_classes_det_averaged``.
- ``trackmap_tracks``: ``_group_tracks`` lengths, mean areas, and mean scores.

Each case stores ``inputs_fnv``, a digest of its generated inputs, so a generator
mismatch fails as such.

It writes ``tests/integration/data/hota_trackmap_internals.json``. The Rust unit tests in
``metrics/hota.rs`` and ``metrics/track_map.rs`` regenerate each input from its seed with
the same generator. Score sets are stored per field as ``[field, shape, fnv]``, where
``fnv`` is FNV-1a 64 over the little-endian ``float64`` bytes (a scalar has shape ``[]``).
Run with the reference interpreter:

    artifacts/rust-port/ref-venv/bin/python scripts/testdata/dump_metric_internals.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
# After site-packages: `moteval` is the pure-Python reference (scripts/testdata/reference.py).
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.testdata.reference import require_reference

require_reference()

import moteval.metrics.hota as hota_module
from moteval import HOTA, Scores, SequenceData, TrackMAP
from moteval.data.model import BoxGeometry
from moteval.metrics.track_map import (
    LABELS,  # pyrefly: ignore[missing-module-attribute]
    _group_tracks,  # pyrefly: ignore[missing-module-attribute]
)
from scripts.testdata.testgen import MASK64, SplitMix64, f64_bits

OUT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "integration"
    / "data"
    / "hota_trackmap_internals.json"
)

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

HOTA_DENSE_CASES = 24
HOTA_COMBINER_CASES = 40
TRACKMAP_COMBINER_CASES = 40
TRACKMAP_TRACK_CASES = 300

Json = Any


def fnv1a64_words(words: list[int]) -> int:
    """FNV-1a over each word's little-endian ``u64`` bytes."""
    h = FNV_OFFSET
    for word in words:
        for byte in word.to_bytes(8, "little"):
            h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def digest_value(value: float | np.ndarray) -> tuple[list[int], int]:
    array = np.asarray(value, dtype=np.float64)
    words = np.ascontiguousarray(array).ravel().view(np.uint64).tolist()
    return list(array.shape), fnv1a64_words(words)


def float_words(values: object) -> list[int]:
    """The ``u64`` bit patterns of ``values`` (a float or an array), in C order."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return array.ravel().view(np.uint64).tolist()


def sequence_words(data: SequenceData) -> list[int]:
    """Per frame: the gt id count and ids, the pred id count and ids, the similarity bits."""
    words: list[int] = []
    for gt, pred, sim in zip(data.gt_ids, data.pred_ids, data.similarity, strict=True):
        words += [len(gt), *gt.tolist(), len(pred), *pred.tolist(), *float_words(sim)]
    return words


def digest_scores(scores: Scores) -> Json:
    """``[[field, shape, fnv], ...]`` in the dict's order."""
    return [[field, *digest_value(value)] for field, value in scores.items()]


def shuffle(rng: SplitMix64, items: list[int]) -> None:
    """Fisher-Yates, from the last position down."""
    for i in range(len(items) - 1, 0, -1):
        j = rng.below(i + 1)
        items[i], items[j] = items[j], items[i]


def frame_ids(rng: SplitMix64, num_ids: int) -> list[int]:
    """A random subset of ``range(num_ids)`` in random order, usually 8 or more ids.

    One frame in eight repeats its first id, which numpy's fancy-index ``+=`` counts once.
    """
    count = 8 + rng.below(num_ids - 7) if rng.below(5) else rng.below(3)
    ids = list(range(num_ids))
    shuffle(rng, ids)
    ids = ids[:count]
    if ids and rng.below(8) == 0:
        ids.append(ids[0])
    return ids


def hota_dense_sequence(rng: SplitMix64, name: str) -> SequenceData:
    num_timesteps = 4 + rng.below(5)
    num_gt_ids = 8 + rng.below(10)
    num_pred_ids = 8 + rng.below(17)
    gt_ids, pred_ids, similarity = [], [], []
    for _ in range(num_timesteps):
        gt = frame_ids(rng, num_gt_ids)
        pred = frame_ids(rng, num_pred_ids)
        sim = [0.0 if rng.below(4) == 0 else rng.uniform() for _ in range(len(gt) * len(pred))]
        gt_ids.append(np.array(gt, dtype=np.int64))
        pred_ids.append(np.array(pred, dtype=np.int64))
        similarity.append(np.array(sim, dtype=np.float64).reshape(len(gt), len(pred)))
    return SequenceData(
        name=name,
        num_timesteps=num_timesteps,
        num_gt_ids=num_gt_ids,
        num_pred_ids=num_pred_ids,
        num_gt_dets=sum(len(ids) for ids in gt_ids),
        num_pred_dets=sum(len(ids) for ids in pred_ids),
        gt_ids=tuple(gt_ids),
        pred_ids=tuple(pred_ids),
        pred_confidences=tuple(np.ones(len(ids)) for ids in pred_ids),
        geometry=BoxGeometry(
            gt=tuple(np.zeros((len(ids), 4)) for ids in gt_ids),
            pred=tuple(np.zeros((len(ids), 4)) for ids in pred_ids),
        ),
        similarity=tuple(similarity),
    )


def hota_dense_case(seed: int) -> Json:
    """Three dense sequences: every LSAP cost matrix, each sequence's scores, combined."""
    rng = SplitMix64(seed)
    sequences = [hota_dense_sequence(rng, f"dense{i}") for i in range(3)]
    costs: list[int] = []
    real_lsap = hota_module.linear_sum_assignment  # pyrefly: ignore[missing-attribute]

    def recording_lsap(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        costs.extend(np.ascontiguousarray(cost, dtype=np.float64).ravel().view(np.uint64).tolist())
        return real_lsap(cost)

    metric = HOTA()
    hota_module.linear_sum_assignment = recording_lsap  # pyrefly: ignore[missing-attribute]
    try:
        per_sequence = [metric.eval_sequence(seq) for seq in sequences]
    finally:
        hota_module.linear_sum_assignment = real_lsap  # pyrefly: ignore[missing-attribute]
    combined = metric.combine_sequences({seq.name: s for seq, s in zip(sequences, per_sequence)})
    return {
        "seed": seed,
        "inputs_fnv": fnv1a64_words([w for seq in sequences for w in sequence_words(seq)]),
        "num_costs": len(costs),
        "costs_fnv": fnv1a64_words(costs),
        "per_sequence": [digest_scores(s) for s in per_sequence],
        "combined": digest_scores(combined),
    }


def hota_class_scores(rng: SplitMix64) -> Scores:
    scores: Scores = {}
    for field in hota_module._FLOAT_ARRAY_FIELDS:  # pyrefly: ignore[missing-attribute]
        scores[field] = np.array([rng.uniform() for _ in range(19)])
    for field in hota_module._INTEGER_ARRAY_FIELDS:  # pyrefly: ignore[missing-attribute]
        scores[field] = np.array([float(rng.below(500)) for _ in range(19)])
    for field in hota_module._FLOAT_FIELDS:  # pyrefly: ignore[missing-attribute]
        scores[field] = rng.uniform()
    return scores


def trackmap_class_scores(rng: SplitMix64) -> Scores:
    scores: Scores = {}
    for label in LABELS:
        for field in (f"AP_{label}", f"AR_{label}"):
            scores[field] = np.array(
                [-1.0 if rng.below(4) == 0 else rng.uniform() for _ in range(10)]
            )
        scores[f"_num_dt_{label}"] = np.array(
            [float(rng.below(4) * rng.below(30)) for _ in range(10)]
        )
    return scores


def combiner_case(seed: int, metric: HOTA | TrackMAP) -> Json:
    rng = SplitMix64(seed)
    num_classes = 8 + rng.below(13)
    make = hota_class_scores if isinstance(metric, HOTA) else trackmap_class_scores
    all_res = {str(c): make(rng) for c in range(num_classes)}
    inputs = [w for res in all_res.values() for v in res.values() for w in float_words(v)]
    return {
        "seed": seed,
        "inputs_fnv": fnv1a64_words(inputs),
        "num_classes": num_classes,
        "class_averaged": digest_scores(metric.combine_classes_class_averaged(all_res)),
        "det_averaged": digest_scores(metric.combine_classes_det_averaged(all_res)),
    }


def trackmap_tracks_case(seed: int) -> Json:
    """Tracks on random frames with random boxes and confidences, ids ascending per frame."""
    rng = SplitMix64(seed)
    num_timesteps = 20 + rng.below(40)
    num_ids = 1 + rng.below(6)
    ids_per_frame, boxes_per_frame, confidences_per_frame = [], [], []
    for _ in range(num_timesteps):
        ids, boxes, confidences = [], [], []
        for track_id in range(num_ids):
            if rng.below(3) == 0:
                continue
            ids.append(track_id)
            boxes.append(
                [
                    rng.uniform() * 50.0,
                    rng.uniform() * 50.0,
                    rng.uniform() * 64.0,
                    rng.uniform() * 64.0,
                ]
            )
            confidences.append(rng.uniform())
        ids_per_frame.append(np.array(ids, dtype=np.int64))
        boxes_per_frame.append(np.array(boxes, dtype=np.float64).reshape(-1, 4))
        confidences_per_frame.append(np.array(confidences, dtype=np.float64))
    _, lengths, areas, scores = _group_tracks(
        tuple(ids_per_frame), tuple(boxes_per_frame), num_ids, tuple(confidences_per_frame)
    )
    assert scores is not None
    inputs: list[int] = []
    for ids, boxes, confidences in zip(
        ids_per_frame, boxes_per_frame, confidences_per_frame, strict=True
    ):
        inputs += [len(ids), *ids.tolist(), *float_words(boxes), *float_words(confidences)]
    return {
        "seed": seed,
        "inputs_fnv": fnv1a64_words(inputs),
        "lengths": lengths,
        "areas": [f64_bits(a) for a in areas],
        "scores": [f64_bits(s) for s in scores],
    }


def main() -> None:
    body = {
        "format": 1,
        "hota_dense": [hota_dense_case(20_000_000 + i) for i in range(HOTA_DENSE_CASES)],
        "hota_class_combiners": [
            combiner_case(21_000_000 + i, HOTA()) for i in range(HOTA_COMBINER_CASES)
        ],
        "trackmap_class_combiners": [
            combiner_case(22_000_000 + i, TrackMAP()) for i in range(TRACKMAP_COMBINER_CASES)
        ],
        "trackmap_tracks": [
            trackmap_tracks_case(23_000_000 + i) for i in range(TRACKMAP_TRACK_CASES)
        ],
    }
    OUT.write_text(json.dumps(body, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
