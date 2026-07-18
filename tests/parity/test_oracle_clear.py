"""Parity tests: moteval CLEAR vs the vendored TrackEval oracle on synthetic fixtures.

Same preprocessing-neutrality choice as tests/parity/test_oracle_synthetic.py: every
fixture passes ``do_preproc=False`` to the oracle, and every GT row here sets
``zero_marked=1`` (the "consider" flag), so both sides score CLEAR on the identical,
unfiltered detections read from the same on-disk fixture files.

Predictions are numbered independently of GT (disjoint id ranges) and frames are read
from the same MOT-txt files the oracle reads, never reconstructed in memory.
"""

import sys
from pathlib import Path

import motmetrics as mm
import numpy as np
import pytest

from moteval import CLEAR, evaluate
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.formats.mot_txt import read_mot
from tests.oracle.runner import run_mot_challenge

# Same sys.path mechanism as tests/oracle/runner.py: makes `_trackeval` importable as
# a top-level package for a direct check of the oracle's own CLEAR combiner methods.
_ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracle"
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

from _trackeval.metrics import CLEAR as OracleCLEAR  # noqa: E402

CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
CLEAR_FIELDS = CLEAR().fields


def _write_rows(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(",".join(str(v) for v in row) for row in rows)
    path.write_text(text + "\n" if rows else "")


def _run(tmp_path: Path, sequences: list[tuple[str, int, list[list], list[list]]]):
    """Build both sides from the same on-disk fixtures and return their CLEAR scores.

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
        name="synthetic", split="val", sequences=tuple(gt_sequences), frame_convention=CONVENTION
    )
    moteval_result = evaluate(dataset, tmp_path / "trackers" / "oracle" / "data", [CLEAR()])
    oracle_result = run_mot_challenge(
        tmp_path / "gt", tmp_path / "trackers", seq_lengths, do_preproc=False, metrics=("CLEAR",)
    )["CLEAR"]
    return moteval_result, oracle_result


def _assert_clear_fields_equal(moteval_scores, oracle_scores) -> None:
    for field in CLEAR_FIELDS:
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
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)


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
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)


def test_id_switches(tmp_path):
    # one continuous gt track, fully detected every frame, but the predicted id
    # changes partway through.
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
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)
    assert moteval_result.combined["CLEAR"]["IDSW"] == 1.0


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
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)


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
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)


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
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)


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
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)


def test_combine_classes_matches_oracle(tmp_path):
    """Class-averaged and det-averaged combiners, checked directly against the oracle.

    Mirrors test_oracle_synthetic.test_combine_classes_matches_oracle: the MOTChallenge
    runner only ever evaluates one class, so this builds two per-sequence CLEAR results
    with deliberately different det/FN counts via moteval's own ``eval_sequence``
    (through ``evaluate``), treats them as two classes, and feeds the exact same dicts
    to both moteval's combiners and the vendored oracle ``CLEAR`` class's combiners.
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
        "class_a": moteval_result.per_sequence["CLASSA01"]["CLEAR"],
        "class_b": moteval_result.per_sequence["CLASSB01"]["CLEAR"],
    }

    moteval_class_avg = CLEAR().combine_classes_class_averaged(all_res)
    moteval_det_avg = CLEAR().combine_classes_det_averaged(all_res)
    oracle_class_avg = OracleCLEAR().combine_classes_class_averaged(all_res)
    oracle_det_avg = OracleCLEAR().combine_classes_det_averaged(all_res)

    _assert_clear_fields_equal(moteval_class_avg, oracle_class_avg)
    _assert_clear_fields_equal(moteval_det_avg, oracle_det_avg)

    # Proves the test has teeth: with genuinely different per-class det/FN counts, a
    # wrong combiner formula would make class-averaged and det-averaged coincide.
    assert moteval_class_avg["MOTA"] != moteval_det_avg["MOTA"]


def test_empty_gt(tmp_path):
    gt_rows: list[list] = []
    pred_rows = [[1, 1, 10, 10, 20, 40, 1]]
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYGT01", 1, gt_rows, pred_rows)])
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)
    assert moteval_result.combined["CLEAR"]["CLR_FP"] == 1.0
    # MLR=1.0 is set on the empty-gt early-return path, but combine_sequences
    # recomputes MLR from the (zero) summed MT/PT/ML counts, so it resets to 0 --
    # check the per-sequence result, where the early-return override still holds.
    assert moteval_result.per_sequence["EMPTYGT01"]["CLEAR"]["MLR"] == 1.0


def test_empty_preds(tmp_path):
    gt_rows = [[1, 1, 10, 10, 20, 40, 1, 1, 1]]
    pred_rows: list[list] = []
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYPRED01", 1, gt_rows, pred_rows)])
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)
    assert moteval_result.combined["CLEAR"]["CLR_FN"] == 1.0
    assert moteval_result.combined["CLEAR"]["ML"] == 1.0
    assert moteval_result.combined["CLEAR"]["MLR"] == 1.0


def test_both_empty(tmp_path):
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYBOTH01", 1, [], [])])
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)
    assert moteval_result.combined["CLEAR"]["CLR_TP"] == 0.0
    assert moteval_result.combined["CLEAR"]["CLR_FP"] == 0.0
    assert moteval_result.combined["CLEAR"]["CLR_FN"] == 0.0


def test_single_frame_track(tmp_path):
    # a gt track that exists for exactly one frame, matched cleanly.
    gt_rows = [[1, 1, 10, 10, 20, 40, 1, 1, 1]]
    pred_rows = [[1, 701, 10, 10, 20, 40, 1]]
    moteval_result, oracle_scores = _run(tmp_path, [("SINGLE01", 1, gt_rows, pred_rows)])
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)
    assert moteval_result.combined["CLEAR"]["CLR_TP"] == 1.0
    assert moteval_result.combined["CLEAR"]["MT"] == 1.0


def test_gap_then_resume_same_id_frag_no_idsw(tmp_path):
    # gt id 1 is present at frames 1, 2, 4 but absent (not a false negative --
    # simply not present in gt) at frame 3. gt id 2 is a constant anchor present
    # and matched every frame so frame 3 is still "considered" (both gt_ids_t and
    # pred_ids_t non-empty), which is what makes gt id 1's gap register at all.
    # Same pred id (901) resumes after the gap -> a fragmentation, but no ID switch.
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [1, 2, 500, 500, 20, 40, 1, 1, 1],
        [2, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 2, 500, 500, 20, 40, 1, 1, 1],
        [3, 2, 500, 500, 20, 40, 1, 1, 1],
        [4, 1, 10, 10, 20, 40, 1, 1, 1],
        [4, 2, 500, 500, 20, 40, 1, 1, 1],
    ]
    pred_rows = [
        [1, 901, 10, 10, 20, 40, 1],
        [1, 950, 500, 500, 20, 40, 1],
        [2, 901, 10, 10, 20, 40, 1],
        [2, 950, 500, 500, 20, 40, 1],
        [3, 950, 500, 500, 20, 40, 1],
        [4, 901, 10, 10, 20, 40, 1],
        [4, 950, 500, 500, 20, 40, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("GAP01", 4, gt_rows, pred_rows)])
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)
    assert moteval_result.combined["CLEAR"]["IDSW"] == 0.0
    assert moteval_result.combined["CLEAR"]["Frag"] == 1.0


def test_gap_then_resume_different_id_frag_and_idsw(tmp_path):
    # identical gap in gt id 1 as above, but the pred id resuming after the gap
    # (902) differs from the one before it (901) -> both a fragmentation and an
    # ID switch.
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [1, 2, 500, 500, 20, 40, 1, 1, 1],
        [2, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 2, 500, 500, 20, 40, 1, 1, 1],
        [3, 2, 500, 500, 20, 40, 1, 1, 1],
        [4, 1, 10, 10, 20, 40, 1, 1, 1],
        [4, 2, 500, 500, 20, 40, 1, 1, 1],
    ]
    pred_rows = [
        [1, 901, 10, 10, 20, 40, 1],
        [1, 950, 500, 500, 20, 40, 1],
        [2, 901, 10, 10, 20, 40, 1],
        [2, 950, 500, 500, 20, 40, 1],
        [3, 950, 500, 500, 20, 40, 1],
        [4, 902, 10, 10, 20, 40, 1],
        [4, 950, 500, 500, 20, 40, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("GAP02", 4, gt_rows, pred_rows)])
    _assert_clear_fields_equal(moteval_result.combined["CLEAR"], oracle_scores)
    assert moteval_result.combined["CLEAR"]["IDSW"] == 1.0
    assert moteval_result.combined["CLEAR"]["Frag"] == 1.0


def test_motmetrics_cross_check_mota_motp_idsw(tmp_path):
    """Independent cross-check against py-motmetrics' mota/motp/num_switches.

    Fixture is unambiguous (no ties, no assignment ambiguity) so both TrackEval's
    global bonus-weighted Hungarian matching and motmetrics' own hysteresis-based
    matching pick the identical pairs every frame, making MOTA and IDSW (num_switches)
    exact matches. motmetrics' MOTP is a mean *distance* (1 - IoU) over matched pairs,
    while TrackEval/moteval's MOTP is a mean *similarity* (IoU); with identical matched
    pairs, ``motmetrics_motp == 1 - trackeval_MOTP`` exactly, so we assert that
    transformed identity rather than raw equality.
    """
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
        [3, 1, 14, 10, 20, 40, 1, 1, 1],
        [4, 1, 16, 10, 20, 40, 1, 1, 1],
        [1, 2, 100, 50, 30, 60, 1, 1, 1],
        [2, 2, 102, 50, 30, 60, 1, 1, 1],
        [3, 2, 104, 50, 30, 60, 1, 1, 1],
        [4, 2, 106, 50, 30, 60, 1, 1, 1],
    ]
    pred_rows = [
        [1, 10, 10, 10, 20, 40, 1],
        [2, 10, 12, 10, 20, 40, 1],
        [3, 11, 14, 10, 20, 40, 1],
        [4, 11, 16, 10, 20, 40, 1],
        [1, 20, 100, 50, 30, 60, 1],
        [2, 20, 102, 50, 30, 60, 1],
        [3, 20, 104, 50, 30, 60, 1],
        [4, 20, 106, 50, 30, 60, 1],
        [2, 99, 500, 500, 10, 10, 1],
    ]
    moteval_result, _ = _run(tmp_path, [("XCHECK01", 4, gt_rows, pred_rows)])
    clear_scores = moteval_result.combined["CLEAR"]

    frames: dict[int, tuple[dict[int, list[float]], dict[int, list[float]]]] = {}
    for row in gt_rows:
        frame, track_id, x, y, w, h = row[:6]
        gt_frame, _ = frames.setdefault(frame, ({}, {}))
        gt_frame[track_id] = [x, y, w, h]
    for row in pred_rows:
        frame, track_id, x, y, w, h = row[:6]
        _, pred_frame = frames.setdefault(frame, ({}, {}))
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
    summary = mh.compute(acc, metrics=["mota", "motp", "num_switches"], name="seq")

    assert clear_scores["MOTA"] == pytest.approx(summary["mota"].iloc[0])
    assert clear_scores["IDSW"] == pytest.approx(summary["num_switches"].iloc[0])
    assert (1.0 - clear_scores["MOTP"]) == pytest.approx(summary["motp"].iloc[0])
