"""Parity tests: moteval TrackMAP vs the vendored TrackEval oracle on synthetic fixtures.

Oracle invocation choice: TrackMAP is not wired into the MOTChallenge2DBox dataset (upstream
ties it to TAO/YouTubeVIS/BURST-style datasets whose loaders precompute track-level areas,
lengths and scores), so `tests/oracle/runner.py`'s ``run_mot_challenge`` cannot produce it.
Instead, every fixture here is a plain Python spec of GT/pred tracks (``{track_id: {frame:
box}}`` for GT, ``{track_id: {frame: (box, confidence)}}`` for predictions), and each test
builds BOTH sides directly from that same spec: a `SequenceData` via `_build_sequence_data`
(moteval's real input) and an upstream-shaped ``data`` dict via `_oracle_data` (fed straight
to the vendored ``TrackMAP`` class, mirroring how the combiner tests in the other parity
suites import oracle classes directly). The two builders are independent implementations of
the same track-level view (areas/lengths as means over each track's own detections, matching
upstream's TAO/BURST dataset convention), so agreement is a genuine cross-check, not a
tautology.

GT/pred track ids are numbered from 1 upward in `_oracle_data` fixtures (never 0): oracle's
own greedy matcher tracks "already matched" via ``gt_m[thr, gt] > 0`` against the literal
matched track id, which silently misbehaves for a legitimate id of 0. Real TrackEval datasets
never emit id 0, so this never fires upstream; moteval's own `TrackMAP` fixes it (see its
module docstring) since `SequenceData` densifies ids starting at 0. `test_id_zero_hazard_fix`
below numbers moteval's own ids from 0 (as real `SequenceData` always does) while offsetting
the oracle-side ids by one, proving the two sides agree in the semantics that upstream
clearly intends, without deliberately triggering oracle's own id-0 bug on that offset copy.
"""

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from moteval.data.model import SequenceData
from moteval.data.similarity import box_iou
from moteval.metrics.track_map import IOU_THRESHOLDS, TrackMAP

_ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracle"
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

from _trackeval.metrics import TrackMAP as OracleTrackMAP  # noqa: E402

TRACKMAP_FIELDS = TrackMAP().fields

GtTracks = dict[int, dict[int, Sequence[float]]]
PredTracks = dict[int, dict[int, tuple[Sequence[float], float]]]


def _build_sequence_data(
    num_timesteps: int, gt_tracks: GtTracks, pred_tracks: PredTracks
) -> SequenceData:
    gt_ids_sorted = sorted(gt_tracks)
    gt_id_map = {tid: i for i, tid in enumerate(gt_ids_sorted)}
    pred_ids_sorted = sorted(pred_tracks)
    pred_id_map = {tid: i for i, tid in enumerate(pred_ids_sorted)}

    gt_ids: list[list[int]] = [[] for _ in range(num_timesteps)]
    gt_boxes: list[list] = [[] for _ in range(num_timesteps)]
    pred_ids: list[list[int]] = [[] for _ in range(num_timesteps)]
    pred_boxes: list[list] = [[] for _ in range(num_timesteps)]
    pred_confs: list[list[float]] = [[] for _ in range(num_timesteps)]

    for tid, frames in gt_tracks.items():
        for t, box in frames.items():
            gt_ids[t].append(gt_id_map[tid])
            gt_boxes[t].append(box)
    for tid, frames in pred_tracks.items():
        for t, (box, conf) in frames.items():
            pred_ids[t].append(pred_id_map[tid])
            pred_boxes[t].append(box)
            pred_confs[t].append(conf)

    gt_ids_arr = tuple(np.array(f, dtype=np.int64) for f in gt_ids)
    gt_boxes_arr = tuple(np.array(f, dtype=np.float64).reshape(-1, 4) for f in gt_boxes)
    pred_ids_arr = tuple(np.array(f, dtype=np.int64) for f in pred_ids)
    pred_boxes_arr = tuple(np.array(f, dtype=np.float64).reshape(-1, 4) for f in pred_boxes)
    pred_confs_arr = tuple(np.array(f, dtype=np.float64) for f in pred_confs)
    similarity = tuple(box_iou(g, p) for g, p in zip(gt_boxes_arr, pred_boxes_arr, strict=True))

    return SequenceData(
        name="synthetic",
        num_timesteps=num_timesteps,
        num_gt_ids=len(gt_ids_sorted),
        num_pred_ids=len(pred_ids_sorted),
        num_gt_dets=sum(a.shape[0] for a in gt_boxes_arr),
        num_pred_dets=sum(a.shape[0] for a in pred_boxes_arr),
        gt_ids=gt_ids_arr,
        pred_ids=pred_ids_arr,
        pred_confidences=pred_confs_arr,
        gt_boxes=gt_boxes_arr,
        pred_boxes=pred_boxes_arr,
        similarity=similarity,
    )


def _oracle_data(gt_tracks: GtTracks, pred_tracks: PredTracks) -> dict:
    gt_ids_sorted = sorted(gt_tracks)
    pred_ids_sorted = sorted(pred_tracks)
    gt_track_dicts = [
        {t: np.array(box) for t, box in gt_tracks[tid].items()} for tid in gt_ids_sorted
    ]
    dt_track_dicts = [
        {t: np.array(box) for t, (box, _) in pred_tracks[tid].items()} for tid in pred_ids_sorted
    ]
    gt_areas = [
        float(np.mean([b[2] * b[3] for b in gt_tracks[tid].values()])) for tid in gt_ids_sorted
    ]
    gt_lengths = [len(gt_tracks[tid]) for tid in gt_ids_sorted]
    dt_areas = [
        float(np.mean([b[2] * b[3] for b, _ in pred_tracks[tid].values()]))
        for tid in pred_ids_sorted
    ]
    dt_lengths = [len(pred_tracks[tid]) for tid in pred_ids_sorted]
    dt_scores = [
        float(np.mean([c for _, c in pred_tracks[tid].values()])) for tid in pred_ids_sorted
    ]

    data = {
        "gt_track_ids": gt_ids_sorted,
        "dt_track_ids": pred_ids_sorted,
        "gt_tracks": gt_track_dicts,
        "dt_tracks": dt_track_dicts,
        "gt_track_areas": gt_areas,
        "gt_track_lengths": gt_lengths,
        "dt_track_areas": dt_areas,
        "dt_track_lengths": dt_lengths,
        "dt_track_scores": np.array(dt_scores),
        "iou_type": "bbox",
        "boxformat": "xywh",
    }
    if data["dt_tracks"]:
        idx = np.argsort([-s for s in dt_scores], kind="mergesort")
        data["dt_track_scores"] = data["dt_track_scores"][idx]
        data["dt_tracks"] = [data["dt_tracks"][i] for i in idx]
        data["dt_track_ids"] = [data["dt_track_ids"][i] for i in idx]
        data["dt_track_areas"] = [data["dt_track_areas"][i] for i in idx]
        data["dt_track_lengths"] = [data["dt_track_lengths"][i] for i in idx]
    return data


def _run(sequences: dict[str, tuple[int, GtTracks, PredTracks]]):
    """Build both sides from the same fixtures and return their combined TrackMAP scores.

    ``sequences`` maps sequence name -> ``(num_timesteps, gt_tracks, pred_tracks)``.
    """
    moteval_res = {}
    oracle_res = {}
    for name, (num_timesteps, gt_tracks, pred_tracks) in sequences.items():
        sd = _build_sequence_data(num_timesteps, gt_tracks, pred_tracks)
        od = _oracle_data(gt_tracks, pred_tracks)
        moteval_res[name] = TrackMAP().eval_sequence(sd)
        oracle_res[name] = OracleTrackMAP().eval_sequence(od)
    moteval_combined = TrackMAP().combine_sequences(moteval_res)
    oracle_combined = OracleTrackMAP().combine_sequences(oracle_res)
    return moteval_combined, oracle_combined


def _assert_trackmap_fields_equal(moteval_scores, oracle_scores) -> None:
    for field in TRACKMAP_FIELDS:
        m = np.asarray(moteval_scores[field])
        o = np.asarray(oracle_scores[field])
        assert np.array_equal(m, o), f"{field} mismatch: moteval={m} oracle={o}"


def test_perfect_match():
    gt_tracks: GtTracks = {1: {t: [10 + t, 10, 20, 20] for t in range(3)}}
    pred_tracks: PredTracks = {101: {t: ([10 + t, 10, 20, 20], 0.9) for t in range(3)}}
    moteval_scores, oracle_scores = _run({"seq": (3, gt_tracks, pred_tracks)})
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)
    # AP is never *exactly* 1.0 -- upstream's precision denominator always adds
    # ``np.spacing(1)`` -- so this is an approx check; `_assert_trackmap_fields_equal`
    # above already proved moteval matches the oracle bit-for-bit on the raw value.
    assert moteval_scores["AP_all"][0] == pytest.approx(1.0)
    assert moteval_scores["AR_all"][0] == 1.0


def test_misses_false_positives_partial_iou_multi_sequence():
    # seq1: gt 1 perfectly matched, gt 2 missed entirely, gt 3 (long, large area)
    # perfectly matched; a short false-positive pred track with no gt counterpart.
    gt_tracks1: GtTracks = {
        1: {t: [10 + t, 10, 20, 20] for t in range(5)},
        2: {t: [200, 200, 15, 15] for t in range(5)},
        3: {t: [400, 400, 300, 300] for t in range(15)},
    }
    pred_tracks1: PredTracks = {
        101: {t: ([10 + t, 10, 20, 20], 0.9) for t in range(5)},
        102: {t: ([600, 600, 10, 10], 0.8) for t in range(3)},
        103: {t: ([400, 400, 300, 300], 0.95) for t in range(15)},
    }
    # seq2: a partial-IoU match (box shifts out of overlap on the second frame).
    gt_tracks2: GtTracks = {10: {0: [50, 50, 50, 50], 1: [52, 50, 50, 50]}}
    pred_tracks2: PredTracks = {201: {0: ([50, 50, 50, 50], 0.7), 1: ([90, 50, 50, 50], 0.7)}}

    moteval_scores, oracle_scores = _run(
        {"seq1": (15, gt_tracks1, pred_tracks1), "seq2": (2, gt_tracks2, pred_tracks2)}
    )
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)
    # Exercises all three area and time subsets simultaneously: track 1 falls in
    # area_s/time_m, track 3 in area_l/time_l, and the FP track in time_s.
    assert moteval_scores["AP_area_m"][0] == 0.0
    assert moteval_scores["AP_area_l"][0] == pytest.approx(1.0)
    assert moteval_scores["AP_time_s"][0] == 0.0
    assert moteval_scores["AP_time_l"][0] == pytest.approx(1.0)


def test_area_boundary_straddle():
    # area_s = [0, 32**2], area_m = [32**2, 96**2]: a track with area exactly 32*32
    # must land in BOTH subsets (upstream's ranges are inclusive on both ends).
    gt_tracks: GtTracks = {1: {t: [0, 0, 32, 32] for t in range(2)}}
    pred_tracks: PredTracks = {101: {t: ([0, 0, 32, 32], 0.9) for t in range(2)}}
    moteval_scores, oracle_scores = _run({"seq": (2, gt_tracks, pred_tracks)})
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)
    assert moteval_scores["AP_area_s"][0] == pytest.approx(1.0)
    assert moteval_scores["AP_area_m"][0] == pytest.approx(1.0)
    assert moteval_scores["AP_area_l"][0] == -1.0


def test_time_boundary_straddle():
    # time_s = [0, 3], time_m = [3, 10]: a track of length exactly 3 lands in both.
    gt_tracks: GtTracks = {1: {t: [10, 10, 5, 5] for t in range(3)}}
    pred_tracks: PredTracks = {101: {t: ([10, 10, 5, 5], 0.9) for t in range(3)}}
    moteval_scores, oracle_scores = _run({"seq": (3, gt_tracks, pred_tracks)})
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)
    assert moteval_scores["AP_time_s"][0] == pytest.approx(1.0)
    assert moteval_scores["AP_time_m"][0] == pytest.approx(1.0)
    assert moteval_scores["AP_time_l"][0] == -1.0


def test_empty_gt():
    pred_tracks: PredTracks = {201: {0: ([1, 1, 1, 1], 0.5)}}
    moteval_scores, oracle_scores = _run({"seq": (2, {}, pred_tracks)})
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)
    assert moteval_scores["AP_all"][0] == -1.0


def test_empty_preds():
    gt_tracks: GtTracks = {1: {0: [1, 1, 1, 1]}}
    moteval_scores, oracle_scores = _run({"seq": (2, gt_tracks, {})})
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)
    assert moteval_scores["AP_all"][0] == 0.0
    assert moteval_scores["AR_all"][0] == 0.0


def test_both_empty():
    moteval_scores, oracle_scores = _run({"seq": (2, {}, {})})
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)
    assert moteval_scores["AP_all"][0] == -1.0
    assert moteval_scores["AR_all"][0] == -1.0


def test_id_zero_hazard_fix():
    """moteval's own densified ids always start at 0 for both gt and pred; the fix
    described in `track_map.py`'s module docstring (boolean matched-state tracking
    instead of upstream's ``gt_m[thr, gt] > 0`` id-value check) must produce the same
    result oracle produces on ids that never hit its own id-0 edge case.
    """
    gt_tracks: GtTracks = {
        0: {t: [10, 10, 20, 20] for t in range(3)},
        1: {t: [10, 10, 20, 20] for t in range(3)},  # identical box: ties with id 0
    }
    pred_tracks: PredTracks = {
        0: {t: ([10, 10, 20, 20], 0.99) for t in range(3)},
        1: {t: ([10, 10, 20, 20], 0.5) for t in range(3)},
    }
    sd = _build_sequence_data(3, gt_tracks, pred_tracks)
    # Oracle side: ids offset by +1 so oracle's own id-0 check never fires either --
    # this compares the intended semantics, not a deliberately-triggered upstream bug.
    gt_tracks_offset = {tid + 1: frames for tid, frames in gt_tracks.items()}
    pred_tracks_offset = {tid + 1: frames for tid, frames in pred_tracks.items()}
    od = _oracle_data(gt_tracks_offset, pred_tracks_offset)

    moteval_scores = TrackMAP().combine_sequences({"seq": TrackMAP().eval_sequence(sd)})
    oracle_scores = OracleTrackMAP().combine_sequences({"seq": OracleTrackMAP().eval_sequence(od)})
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)


def test_dt_track_order_matches_upstream_score_presort():
    """Upstream's dataset loaders (e.g. tao.py's `get_preprocessed_seq_data`) sort dt
    tracks by descending confidence (mergesort, so score ties keep original order)
    BEFORE eval_sequence ever runs. The per-threshold greedy match is order-dependent
    on IoU ties (whichever dt is considered first wins a contested gt), so `eval_sequence`
    must presort dt tracks the same way `_oracle_data` already does for this suite's
    oracle side, not just rely on `combine_sequences`' later global score resort (that
    resort only reorders the *ranking* used for precision/recall, after TP/FP
    assignment has already been fixed by the greedy match).

    Fixture: one gt track, two dt tracks both perfectly overlapping it (an IoU tie), so
    only one can match. The LOW original/densified id gets the LOW score and the HIGH
    id gets the HIGH score -- if dt processing order followed id order instead of
    descending score, the wrong (low-score) dt would win the match and AP_all would
    drop to 0.5 instead of ~1.0.
    """
    gt_tracks: GtTracks = {5: {t: [10, 10, 20, 20] for t in range(3)}}
    pred_tracks: PredTracks = {
        1: {t: ([10, 10, 20, 20], 0.5) for t in range(3)},  # low score, low id
        9: {t: ([10, 10, 20, 20], 0.9) for t in range(3)},  # high score, high id
    }
    moteval_scores, oracle_scores = _run({"seq": (3, gt_tracks, pred_tracks)})
    _assert_trackmap_fields_equal(moteval_scores, oracle_scores)
    assert moteval_scores["AP_all"][0] == pytest.approx(1.0)


def test_combine_classes_class_averaged_matches_oracle():
    """Class-averaged combiner, checked directly against the oracle.

    Mirrors test_oracle_identity.test_combine_classes_matches_oracle: builds two
    per-class combined TrackMAP results (via moteval's own eval_sequence +
    combine_sequences on two distinct fixtures) and feeds the exact same dict to
    both moteval's and the vendored oracle TrackMAP class's class-averaged combiner.
    """
    class_a_gt: GtTracks = {1: {t: [10 + t, 10, 20, 20] for t in range(3)}}
    class_a_pred: PredTracks = {101: {t: ([10 + t, 10, 20, 20], 0.9) for t in range(3)}}
    class_b_gt: GtTracks = {
        1: {t: [10 + t, 10, 20, 20] for t in range(5)},
        2: {t: [200, 200, 15, 15] for t in range(5)},
    }
    class_b_pred: PredTracks = {101: {t: ([10 + t, 10, 20, 20], 0.9) for t in range(5)}}

    class_a = TrackMAP().combine_sequences(
        {"seq": TrackMAP().eval_sequence(_build_sequence_data(3, class_a_gt, class_a_pred))}
    )
    class_b = TrackMAP().combine_sequences(
        {"seq": TrackMAP().eval_sequence(_build_sequence_data(5, class_b_gt, class_b_pred))}
    )
    all_res = {"class_a": class_a, "class_b": class_b}

    moteval_class_avg = TrackMAP().combine_classes_class_averaged(all_res)
    oracle_class_avg = OracleTrackMAP().combine_classes_class_averaged(all_res)
    _assert_trackmap_fields_equal(moteval_class_avg, oracle_class_avg)


def test_combine_classes_det_averaged_hand_computed_divergence():
    """Documented divergence: upstream's `combine_classes_det_averaged` is a
    copy-paste of the class-averaged combiner (never weights by detections).
    moteval instead weights each class by `_num_dt_<lbl>`. Verifies moteval's
    output against a hand-computed weighted average, and that it differs from
    the oracle's (buggy, unweighted) output on the same input.
    """
    # class_a: a single perfectly-matched track (weight 1 detection, AP ~= 1.0).
    class_a_gt: GtTracks = {1: {t: [10 + t, 10, 20, 20] for t in range(3)}}
    class_a_pred: PredTracks = {101: {t: ([10 + t, 10, 20, 20], 0.9) for t in range(3)}}
    # class_b: two perfectly-matched tracks plus two false-positive pred tracks
    # RANKED IN BETWEEN them by confidence (weight 4 detections, AP well below
    # 1.0 since the FPs degrade precision before full recall is reached) -- both
    # the weight and the AP value genuinely differ from class_a, so a weighted
    # vs. unweighted mean diverge.
    class_b_gt: GtTracks = {
        1: {t: [10 + t, 10, 20, 20] for t in range(5)},
        2: {t: [200, 200, 15, 15] for t in range(5)},
    }
    class_b_pred: PredTracks = {
        101: {t: ([10 + t, 10, 20, 20], 0.99) for t in range(5)},
        103: {t: ([600, 600, 10, 10], 0.97) for t in range(5)},
        104: {t: ([700, 700, 10, 10], 0.95) for t in range(5)},
        102: {t: ([200, 200, 15, 15], 0.5) for t in range(5)},
    }

    class_a = TrackMAP().combine_sequences(
        {"seq": TrackMAP().eval_sequence(_build_sequence_data(3, class_a_gt, class_a_pred))}
    )
    class_b = TrackMAP().combine_sequences(
        {"seq": TrackMAP().eval_sequence(_build_sequence_data(5, class_b_gt, class_b_pred))}
    )
    all_res = {"class_a": class_a, "class_b": class_b}

    moteval_det_avg = TrackMAP().combine_classes_det_averaged(all_res)
    oracle_det_avg = OracleTrackMAP().combine_classes_det_averaged(all_res)

    # Independently counted from the raw fixture (not read from combine_sequences'
    # `_num_dt_all` output, which would make this a tautology instead of a check): the
    # "all" label never ignores any detection (its ignore mask is all-zero, by area
    # and time range alike), so its weight is simply each class's total pred-track
    # count, constant across every IoU threshold.
    w_a = np.full(len(IOU_THRESHOLDS), len(class_a_pred), dtype=float)
    w_b = np.full(len(IOU_THRESHOLDS), len(class_b_pred), dtype=float)
    v_a, v_b = class_a["AP_all"], class_b["AP_all"]
    weight_sum = w_a + w_b
    expected_ap_all = np.where(
        weight_sum > 0, (w_a * v_a + w_b * v_b) / weight_sum, (v_a + v_b) / 2
    )

    assert np.allclose(moteval_det_avg["AP_all"], expected_ap_all)
    # Proves the test has teeth: class_b has more (correctly matched) detections
    # than class_a, so the weighted and unweighted means genuinely differ, and
    # moteval's det-averaged result differs from the oracle's buggy one.
    assert not np.array_equal(moteval_det_avg["AP_all"], oracle_det_avg["AP_all"])
    assert not np.array_equal(moteval_det_avg["AP_all"], class_a["AP_all"])
