"""Parity tests: moteval Identity vs the vendored TrackEval oracle on synthetic fixtures.

Same preprocessing-neutrality choice as tests/parity/test_oracle_synthetic.py: every
fixture passes ``do_preproc=False`` to the oracle, and every GT row here sets
``zero_marked=1`` (the "consider" flag), so both sides score Identity on the identical,
unfiltered detections read from the same on-disk fixture files.

Predictions are numbered independently of GT (disjoint id ranges) and frames are read
from the same MOT-txt files the oracle reads, never reconstructed in memory.
"""

import sys
from pathlib import Path

import motmetrics as mm
import numpy as np
import pytest

from moteval import Identity, evaluate
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import read_mot
from tests.oracle.runner import run_mot_challenge

# Same sys.path mechanism as tests/oracle/runner.py: makes `_trackeval` importable as
# a top-level package for a direct check of the oracle's own Identity combiner methods.
_ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracle"
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

from _trackeval.metrics import Identity as OracleIdentity  # noqa: E402

CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
PROTOCOL = Protocol(name="parity", frame_convention=CONVENTION, eval_classes=(1,))
IDENTITY_FIELDS = Identity().fields


def _write_rows(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(",".join(str(v) for v in row) for row in rows)
    path.write_text(text + "\n" if rows else "")


def _run(tmp_path: Path, sequences: list[tuple[str, int, list[list], list[list]]]):
    """Build both sides from the same on-disk fixtures and return their Identity scores.

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
    moteval_result = evaluate(dataset, tmp_path / "trackers" / "oracle" / "data", [Identity()])
    oracle_result = run_mot_challenge(
        tmp_path / "gt", tmp_path / "trackers", seq_lengths, do_preproc=False, metrics=("Identity",)
    )["Identity"]
    return moteval_result, oracle_result


def _assert_identity_fields_equal(moteval_scores, oracle_scores) -> None:
    for field in IDENTITY_FIELDS:
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
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)
    assert moteval_result.combined["Identity"]["IDFN"] == 3.0


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
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)
    assert moteval_result.combined["Identity"]["IDFP"] == 2.0


def test_id_switches(tmp_path):
    # one continuous gt track, fully detected every frame, but the predicted id
    # changes partway through -- the global matcher picks the id used most, so the
    # other half of the track's detections show up as IDFN/IDFP rather than a clean match.
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
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)
    assert moteval_result.combined["Identity"]["IDFN"] == 2.0
    assert moteval_result.combined["Identity"]["IDFP"] == 2.0


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
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)


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
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)


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
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)


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
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)


def test_more_pred_tracks_than_gt_tracks(tmp_path):
    # block matrix shape asymmetry: 1 gt track, 3 pred tracks (2 of them pure FPs).
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
    ]
    pred_rows = [
        [1, 801, 10, 10, 20, 40, 1],
        [2, 801, 12, 10, 20, 40, 1],
        [1, 802, 500, 500, 10, 10, 1],
        [2, 803, 600, 600, 10, 10, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("MOREPRED01", 2, gt_rows, pred_rows)])
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)
    assert moteval_result.combined["Identity"]["IDFP"] == 2.0


def test_more_gt_tracks_than_pred_tracks(tmp_path):
    # block matrix shape asymmetry: 3 gt tracks, 1 pred track (2 gt tracks pure FNs).
    gt_rows = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [1, 2, 500, 500, 10, 10, 1, 1, 1],
        [1, 3, 600, 600, 10, 10, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
        [2, 2, 500, 500, 10, 10, 1, 1, 1],
        [2, 3, 600, 600, 10, 10, 1, 1, 1],
    ]
    pred_rows = [
        [1, 901, 10, 10, 20, 40, 1],
        [2, 901, 12, 10, 20, 40, 1],
    ]
    moteval_result, oracle_scores = _run(tmp_path, [("MOREGT01", 2, gt_rows, pred_rows)])
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)
    assert moteval_result.combined["Identity"]["IDFN"] == 4.0


def test_combine_classes_matches_oracle(tmp_path):
    """Class-averaged and det-averaged combiners, checked directly against the oracle.

    Mirrors test_oracle_clear.test_combine_classes_matches_oracle: the MOTChallenge
    runner only ever evaluates one class, so this builds two per-sequence Identity
    results with deliberately different det/FN counts via moteval's own
    ``eval_sequence`` (through ``evaluate``), treats them as two classes, and feeds
    the exact same dicts to both moteval's combiners and the vendored oracle
    ``Identity`` class's combiners.
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
        "class_a": moteval_result.per_sequence["CLASSA01"]["Identity"],
        "class_b": moteval_result.per_sequence["CLASSB01"]["Identity"],
    }

    moteval_class_avg = Identity().combine_classes_class_averaged(all_res)
    moteval_det_avg = Identity().combine_classes_det_averaged(all_res)
    oracle_class_avg = OracleIdentity().combine_classes_class_averaged(all_res)
    oracle_det_avg = OracleIdentity().combine_classes_det_averaged(all_res)

    _assert_identity_fields_equal(moteval_class_avg, oracle_class_avg)
    _assert_identity_fields_equal(moteval_det_avg, oracle_det_avg)

    # Proves the test has teeth: with genuinely different per-class det/FN counts, a
    # wrong combiner formula would make class-averaged and det-averaged coincide.
    assert moteval_class_avg["IDF1"] != moteval_det_avg["IDF1"]


def test_empty_gt(tmp_path):
    gt_rows: list[list] = []
    pred_rows = [[1, 1, 10, 10, 20, 40, 1]]
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYGT01", 1, gt_rows, pred_rows)])
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)
    assert moteval_result.combined["Identity"]["IDFP"] == 1.0
    # IDF1/IDR/IDP are 0.0 on the empty-gt early-return path (never recomputed via
    # _compute_final_fields), and combine_sequences recomputes them from the
    # (zero IDTP, zero IDFN, one IDFP) summed counts, which yields the same 0.0 --
    # unlike CLEAR's MLR, the early-return quirk is unobservable after combining,
    # so this assertion pins the per-sequence value rather than distinguishing paths.
    assert moteval_result.per_sequence["EMPTYGT01"]["Identity"]["IDF1"] == 0.0


def test_empty_preds(tmp_path):
    gt_rows = [[1, 1, 10, 10, 20, 40, 1, 1, 1]]
    pred_rows: list[list] = []
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYPRED01", 1, gt_rows, pred_rows)])
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)
    assert moteval_result.combined["Identity"]["IDFN"] == 1.0
    assert moteval_result.per_sequence["EMPTYPRED01"]["Identity"]["IDF1"] == 0.0


def test_both_empty(tmp_path):
    moteval_result, oracle_scores = _run(tmp_path, [("EMPTYBOTH01", 1, [], [])])
    _assert_identity_fields_equal(moteval_result.combined["Identity"], oracle_scores)
    assert moteval_result.combined["Identity"]["IDTP"] == 0.0
    assert moteval_result.combined["Identity"]["IDFP"] == 0.0
    assert moteval_result.combined["Identity"]["IDFN"] == 0.0


def test_motmetrics_cross_check_idf1_idr_idp(tmp_path):
    """Independent cross-check against py-motmetrics' idf1/idr/idp/idtp/idfn/idfp.

    Unlike CLEAR (frame-level matching, so its cross-check needs `pytest.approx`),
    py-motmetrics' ID metrics implement the same global bipartite matching (Ristani
    et al.) that TrackEval's Identity metric does, so on an unambiguous fixture (no
    ties, no assignment ambiguity) every field is bit-identical, not just close.
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
    identity_scores = moteval_result.combined["Identity"]

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
    summary = mh.compute(acc, metrics=["idf1", "idr", "idp", "idtp", "idfn", "idfp"], name="seq")

    assert identity_scores["IDF1"] == pytest.approx(summary["idf1"].iloc[0])
    assert identity_scores["IDR"] == pytest.approx(summary["idr"].iloc[0])
    assert identity_scores["IDP"] == pytest.approx(summary["idp"].iloc[0])
    assert identity_scores["IDTP"] == pytest.approx(summary["idtp"].iloc[0])
    assert identity_scores["IDFN"] == pytest.approx(summary["idfn"].iloc[0])
    assert identity_scores["IDFP"] == pytest.approx(summary["idfp"].iloc[0])
