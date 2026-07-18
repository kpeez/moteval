"""Parity tests: moteval HOTA vs the vendored TrackEval oracle on synthetic fixtures.

Preprocessing-neutrality choice: every fixture passes ``do_preproc=False`` to the
oracle. TrackEval's MOT preprocessing (distractor-class removal) only runs when
``do_preproc`` is True; with it False, the oracle's only remaining GT filter is the
``zero_marked`` ("consider") flag, and every GT row here sets it to 1. moteval's
tracer-bullet path has no preprocessing wired in at all, so ``do_preproc=False`` keeps
both sides scoring HOTA on the identical, unfiltered detections read from the same
on-disk fixture files.

Predictions are numbered independently of GT (disjoint id ranges) and frames are read
from the same MOT-txt files the oracle reads, never reconstructed in memory.
"""

import sys
from pathlib import Path

import motmetrics as mm
import numpy as np
import pytest

from moteval import HOTA, evaluate
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import read_mot
from tests.oracle.runner import run_mot_challenge

# Same sys.path mechanism as tests/oracle/runner.py: makes `_trackeval` importable as
# a top-level package for a direct check of the oracle's own HOTA combiner methods.
_ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracle"
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

from _trackeval.metrics import HOTA as OracleHOTA  # noqa: E402

CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
PROTOCOL = Protocol(name="parity", frame_convention=CONVENTION, eval_classes=(1,))
HOTA_FIELDS = HOTA().fields


def _write_rows(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(",".join(str(v) for v in row) for row in rows)
    path.write_text(text + "\n" if rows else "")


def _run(tmp_path: Path, sequences: list[tuple[str, int, list[list], list[list]]]):
    """Build both sides from the same on-disk fixtures and return their HOTA scores.

    ``sequences`` is a list of ``(name, num_timesteps, gt_rows, pred_rows)``.
    """
    gt_sequences = []
    seq_lengths = {}
    for name, num_timesteps, gt_rows, pred_rows in sequences:
        gt_path = tmp_path / "gt" / name / "gt" / "gt.txt"
        pred_path = tmp_path / "trackers" / "oracle" / "data" / f"{name}.txt"
        _write_rows(gt_path, gt_rows)
        _write_rows(pred_path, pred_rows)
        gt_tracks = tuple(read_mot(gt_path))
        gt_sequences.append(GtSequence(name=name, num_timesteps=num_timesteps, tracks=gt_tracks))
        seq_lengths[name] = num_timesteps

    dataset = MOTDataset(
        name="synthetic", split="val", sequences=tuple(gt_sequences), protocol=PROTOCOL
    )
    moteval_result = evaluate(dataset, tmp_path / "trackers" / "oracle" / "data", [HOTA()])
    oracle_result = run_mot_challenge(
        tmp_path / "gt", tmp_path / "trackers", seq_lengths, do_preproc=False, metrics=("HOTA",)
    )["HOTA"]
    return moteval_result, oracle_result


def _assert_hota_fields_equal(moteval_scores, oracle_scores) -> None:
    for field in HOTA_FIELDS:
        m = np.asarray(moteval_scores[field])
        o = np.asarray(oracle_scores[field])
        assert np.array_equal(m, o), f"{field} mismatch: moteval={m} oracle={o}"


def test_misses(tmp_path):
    # gt id 2 is never predicted at all: pure false negatives across every frame.
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
        [3, 1, 14, 10, 20, 40, 1, 1, 1],
        [1, 2, 100, 50, 30, 60, 1, 1, 1],
        [2, 2, 102, 50, 30, 60, 1, 1, 1],
        [3, 2, 104, 50, 30, 60, 1, 1, 1],
    ]
    pred_rows = [
        [1, 101, 10, 10, 20, 40, 1],
        [2, 101, 12, 10, 20, 40, 1],
        [3, 101, 14, 10, 20, 40, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("MISS01", 3, gt_rows, pred_rows)])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)


def test_false_positives(tmp_path):
    # every gt det matches; extra pred dets have no gt counterpart at all.
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
    ]
    pred_rows = [
        [1, 201, 10, 10, 20, 40, 1],
        [2, 201, 12, 10, 20, 40, 1],
        [1, 202, 500, 500, 10, 10, 1],
        [2, 202, 500, 500, 10, 10, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("FP01", 2, gt_rows, pred_rows)])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)


def test_id_switches(tmp_path):
    # one continuous gt track, fully detected every frame, but the predicted id
    # changes partway through: full DetA, degraded AssA.
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
        [3, 1, 14, 10, 20, 40, 1, 1, 1],
        [4, 1, 16, 10, 20, 40, 1, 1, 1],
    ]
    pred_rows = [
        [1, 301, 10, 10, 20, 40, 1],
        [2, 301, 12, 10, 20, 40, 1],
        [3, 302, 14, 10, 20, 40, 1],
        [4, 302, 16, 10, 20, 40, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("IDSW01", 4, gt_rows, pred_rows)])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)


def test_ragged_frames(tmp_path):
    # frame counts vary independently on the gt and pred side per frame.
    gt_rows = [
        [1, 1, 0, 0, 10, 10, 1, 1, 1],
        [2, 1, 0, 0, 10, 10, 1, 1, 1],
        [2, 2, 50, 50, 10, 10, 1, 1, 1],
        [2, 3, 90, 10, 10, 10, 1, 1, 1],
        [4, 1, 0, 0, 10, 10, 1, 1, 1],
        [4, 4, 200, 200, 10, 10, 1, 1, 1],
    ]
    pred_rows = [
        [1, 401, 0, 0, 10, 10, 1],
        [2, 401, 0, 0, 10, 10, 1],
        [2, 402, 50, 50, 10, 10, 1],
        [3, 403, 500, 500, 10, 10, 1],
        [4, 401, 0, 0, 10, 10, 1],
        [4, 404, 200, 200, 10, 10, 1],
        [4, 405, 300, 300, 10, 10, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("RAGGED01", 4, gt_rows, pred_rows)])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)


def test_empty_frames(tmp_path):
    # frames 2 and 4 have no gt and no pred at all.
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [3, 1, 14, 10, 20, 40, 1, 1, 1],
        [5, 1, 18, 10, 20, 40, 1, 1, 1],
    ]
    pred_rows = [
        [1, 501, 10, 10, 20, 40, 1],
        [3, 501, 14, 10, 20, 40, 1],
        [5, 501, 18, 10, 20, 40, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYF01", 5, gt_rows, pred_rows)])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)


def test_tie_heavy_assignment(tmp_path):
    # three gt boxes and three pred boxes stacked at identical coordinates: every
    # gt/pred pair has IoU 1.0, forcing scipy to break ties among equal-cost matches.
    gt_rows = [
        [1, 1, 10, 10, 20, 20, 1, 1, 1],
        [1, 2, 10, 10, 20, 20, 1, 1, 1],
        [1, 3, 10, 10, 20, 20, 1, 1, 1],
    ]
    pred_rows = [
        [1, 601, 10, 10, 20, 20, 1],
        [1, 602, 10, 10, 20, 20, 1],
        [1, 603, 10, 10, 20, 20, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("TIE01", 1, gt_rows, pred_rows)])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)


def test_multi_sequence_combine(tmp_path):
    miss_gt = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
        [1, 2, 100, 50, 30, 60, 1, 1, 1],
        [2, 2, 102, 50, 30, 60, 1, 1, 1],
    ]
    miss_pred = [
        [1, 101, 10, 10, 20, 40, 1],
        [2, 101, 12, 10, 20, 40, 1],
    ]
    fp_gt = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
    ]
    fp_pred = [
        [1, 201, 10, 10, 20, 40, 1],
        [2, 201, 12, 10, 20, 40, 1],
        [1, 202, 500, 500, 10, 10, 1],
        [2, 202, 500, 500, 10, 10, 1],
    ]
    moteval_result, oracle_scores = _run(
        tmp_path,
        [
            ("MISS01", 2, miss_gt, miss_pred),
            ("FP01", 2, fp_gt, fp_pred),
        ],
    )
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)


def test_combine_classes_matches_oracle(tmp_path):
    """Class-averaged and det-averaged combiners, checked directly against the oracle.

    The MOTChallenge runner only ever evaluates one class, so it can't exercise
    ``combine_classes_class_averaged``/``combine_classes_det_averaged`` meaningfully:
    averaging or summing over a single class is a no-op regardless of formula
    correctness. This builds two per-sequence HOTA results with deliberately different
    det counts via moteval's own ``eval_sequence`` (through ``evaluate``), treats them
    as two classes, and feeds the exact same dicts to both moteval's combiners and the
    vendored oracle ``HOTA`` class's combiners.
    """
    class_a_gt = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
    ]
    class_a_pred = [
        [1, 1001, 10, 10, 20, 40, 1],
        [2, 1001, 12, 10, 20, 40, 1],
    ]
    class_b_gt = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
        [3, 1, 14, 10, 20, 40, 1, 1, 1],
        [1, 2, 100, 50, 30, 60, 1, 1, 1],
        [2, 2, 102, 50, 30, 60, 1, 1, 1],
        [3, 2, 104, 50, 30, 60, 1, 1, 1],
    ]
    class_b_pred = [
        [1, 2001, 10, 10, 20, 40, 1],
        [2, 2001, 12, 10, 20, 40, 1],
        [3, 2001, 14, 10, 20, 40, 1],
        [1, 2002, 500, 500, 10, 10, 1],
    ]
    moteval_result, _ = _run(
        tmp_path,
        [
            ("CLASSA01", 2, class_a_gt, class_a_pred),
            ("CLASSB01", 3, class_b_gt, class_b_pred),
        ],
    )
    all_res = {
        "class_a": moteval_result.per_sequence["CLASSA01"]["HOTA"],
        "class_b": moteval_result.per_sequence["CLASSB01"]["HOTA"],
    }

    moteval_class_avg = HOTA().combine_classes_class_averaged(all_res)
    moteval_det_avg = HOTA().combine_classes_det_averaged(all_res)
    oracle_class_avg = OracleHOTA().combine_classes_class_averaged(all_res)
    oracle_det_avg = OracleHOTA().combine_classes_det_averaged(all_res)

    _assert_hota_fields_equal(moteval_class_avg, oracle_class_avg)
    _assert_hota_fields_equal(moteval_det_avg, oracle_det_avg)

    # Proves the test has teeth: with genuinely different per-class det counts, a wrong
    # combiner formula would make class-averaged and det-averaged coincide.
    assert not np.array_equal(moteval_class_avg["DetA"], moteval_det_avg["DetA"])


def test_empty_gt(tmp_path):
    gt_rows: list[list] = []
    pred_rows = [[1, 1, 10, 10, 20, 40, 1]]
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYGT01", 1, gt_rows, pred_rows)])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)
    assert moteval_result.combined["HOTA"]["HOTA_FP"][0] == 1.0
    assert moteval_result.combined["HOTA"]["LocA(0)"] == 1.0


def test_empty_preds(tmp_path):
    gt_rows = [[1, 1, 10, 10, 20, 40, 1, 1, 1]]
    pred_rows: list[list] = []
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYPRED01", 1, gt_rows, pred_rows)])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)
    assert moteval_result.combined["HOTA"]["HOTA_FN"][0] == 1.0
    assert moteval_result.combined["HOTA"]["LocA(0)"] == 1.0


def test_both_empty(tmp_path):
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYBOTH01", 1, [], [])])
    _assert_hota_fields_equal(moteval_result.combined["HOTA"], oracle_scores)
    assert moteval_result.combined["HOTA"]["HOTA_FN"][0] == 0.0
    assert moteval_result.combined["HOTA"]["HOTA_FP"][0] == 0.0
    assert moteval_result.combined["HOTA"]["LocA(0)"] == 1.0


def test_motmetrics_cross_check_hota_alpha(tmp_path):
    """Independent cross-check against py-motmetrics' hota_alpha at alpha=0.5.

    motmetrics matches per-frame on a hard IoU>=0.5 threshold (no TrackEval-style
    global-alignment weighting), so exact equality isn't guaranteed in general — but
    on an unambiguous fixture (no ties, no id-switch-driven association ambiguity)
    both algorithms pick the same matches, so the headline numbers agree.
    """
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
        [3, 1, 14, 10, 20, 40, 1, 1, 1],
        [1, 2, 100, 50, 30, 60, 1, 1, 1],
        [2, 2, 102, 50, 30, 60, 1, 1, 1],
        [3, 2, 104, 50, 30, 60, 1, 1, 1],
    ]
    pred_rows = [
        [1, 10, 10, 10, 20, 40, 1],
        [2, 10, 12, 10, 20, 40, 1],
        [1, 20, 100, 50, 30, 60, 1],
        [2, 21, 102, 50, 30, 60, 1],
        [3, 21, 104, 50, 30, 60, 1],
        [2, 99, 500, 500, 10, 10, 1],
    ]
    moteval_result, _ = _run(tmp_path, [("XCHECK01", 3, gt_rows, pred_rows)])
    # ALPHAS[9] is exactly 0.5 (np.arange(0.05, 0.99, 0.05)).
    moteval_hota_alpha_05 = moteval_result.combined["HOTA"]["HOTA"][9]

    frames: dict[int, tuple[dict[int, list[float]], dict[int, list[float]]]] = {}
    for row in gt_rows:
        frame, track_id, x, y, w, h = row[:6]
        gt_frame, pred_frame = frames.setdefault(frame, ({}, {}))
        gt_frame[track_id] = [x, y, w, h]
    for row in pred_rows:
        frame, track_id, x, y, w, h = row[:6]
        gt_frame, pred_frame = frames.setdefault(frame, ({}, {}))
        pred_frame[track_id] = [x, y, w, h]

    acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(frames):
        gt_frame, pred_frame = frames[frame]
        gt_ids = list(gt_frame)
        pred_ids = list(pred_frame)
        gt_boxes = np.array([gt_frame[i] for i in gt_ids]) if gt_ids else np.empty((0, 4))
        pred_boxes = np.array([pred_frame[i] for i in pred_ids]) if pred_ids else np.empty((0, 4))
        dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, dists, frameid=frame)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["hota_alpha"], name="seq")
    motmetrics_hota_alpha_05 = summary["hota_alpha"].iloc[0]

    assert moteval_hota_alpha_05 == pytest.approx(motmetrics_hota_alpha_05)
