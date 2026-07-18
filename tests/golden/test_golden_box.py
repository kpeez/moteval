"""Golden fixture suite: hand-derived expected values for every box metric.

Independent of the vendored oracle: every expected value below is a hand-derived
constant, guarding against moteval and the oracle sharing a bug. Scenarios and
headline values are ported from track-zoo `f48f449:tests/test_metrics.py`
(perfect / half-missed / ID switch / conf-zero / multi-sequence combine); the
remaining fields and the empty-GT / empty-preds scenarios are derived by hand
from the metric definitions — each scenario documents its derivation inline.

Every fixture runs through the full public path: `GtSequence` -> `MOTDataset`
-> `evaluate()` (which reads predictions from a `<seq>.txt` directory, runs the
protocol engine, densifies ids, and combines sequences). No metric internals
are called. Geometry uses exactly-overlapping or disjoint boxes so IoU is
exactly 1.0 or 0.0 and every expected value is exact in float64 — assertions
are exact equality, never approx.
"""

import numpy as np

from moteval import CLEAR, HOTA, Count, Identity, MOTDataset, Protocol, evaluate
from moteval.data.model import FrameConvention, GtSequence
from moteval.formats.mot_txt import Track, write_mot

GOLDEN_PROTOCOL = Protocol(
    name="golden",
    frame_convention=FrameConvention(name="1-indexed", first_frame=1),
    eval_classes=(1,),
)
METRICS = (HOTA, CLEAR, Identity, Count)
SQRT_HALF = np.sqrt(0.5)


def _boxes(track_id: int, x0: float, frames: int, conf: float = 1.0) -> list[Track]:
    # One 20x20 box per frame, drifting in x. Distinct objects sit >=500px apart
    # so cross-object IoU is exactly 0; identical rows have IoU exactly 1.
    return [
        Track(frame=f, track_id=track_id, x=x0 + f, y=10.0, w=20.0, h=20.0, conf=conf)
        for f in range(1, frames + 1)
    ]


def _dataset(*sequences: GtSequence) -> MOTDataset:
    return MOTDataset(name="golden", split="val", sequences=sequences, protocol=GOLDEN_PROTOCOL)


def _evaluate(dataset: MOTDataset, preds: dict[str, list[Track]], tmp_path):
    for seq_name, tracks in preds.items():
        write_mot(tmp_path / f"{seq_name}.txt", tracks)
    return evaluate(dataset, tmp_path, [metric() for metric in METRICS])


def _alphas(value: float) -> np.ndarray:
    return np.full(19, float(value))


def assert_scores_exact(actual, expected: dict) -> None:
    assert set(actual) == set(expected)
    for field, exp in expected.items():
        act = actual[field]
        if isinstance(exp, np.ndarray):
            assert np.array_equal(act, exp), f"{field}: {act} != {exp}"
        else:
            assert act == exp, f"{field}: {act} != {exp}"


def _hota_expected(
    *,
    tp: float,
    fn: float,
    fp: float,
    det_a: float,
    det_re: float,
    det_pr: float,
    ass_a: float,
    ass_re: float,
    ass_pr: float,
    loca: float,
) -> dict:
    hota = np.sqrt(det_a * ass_a)
    owta = np.sqrt(det_re * ass_a)
    return {
        "HOTA": _alphas(hota),
        "DetA": _alphas(det_a),
        "AssA": _alphas(ass_a),
        "DetRe": _alphas(det_re),
        "DetPr": _alphas(det_pr),
        "AssRe": _alphas(ass_re),
        "AssPr": _alphas(ass_pr),
        "LocA": _alphas(loca),
        "OWTA": _alphas(owta),
        "HOTA_TP": _alphas(tp),
        "HOTA_FN": _alphas(fn),
        "HOTA_FP": _alphas(fp),
        "HOTA(0)": hota,
        "LocA(0)": loca,
        "HOTALocA(0)": hota * loca,
    }


def _clear_expected(
    *,
    tp: float,
    fn: float,
    fp: float,
    idsw: float,
    mt: float,
    pt: float,
    ml: float,
    frag: float,
    frames: float,
    motal: float,
) -> dict:
    # All matches have IoU 1.0, so MOTP_sum == TP and MOTP == 1 whenever TP > 0.
    num_gt_ids = mt + pt + ml
    gt_dets = tp + fn
    return {
        "CLR_TP": tp,
        "CLR_FN": fn,
        "CLR_FP": fp,
        "IDSW": idsw,
        "MT": mt,
        "PT": pt,
        "ML": ml,
        "Frag": frag,
        "CLR_Frames": frames,
        "MOTP_sum": tp,
        "MOTA": (tp - fp - idsw) / max(1.0, gt_dets),
        "MOTP": tp / max(1.0, tp),
        "MODA": (tp - fp) / max(1.0, gt_dets),
        "CLR_Re": tp / max(1.0, gt_dets),
        "CLR_Pr": tp / max(1.0, tp + fp),
        "MTR": mt / max(1.0, num_gt_ids),
        "PTR": pt / max(1.0, num_gt_ids),
        "MLR": ml / max(1.0, num_gt_ids),
        "sMOTA": (tp - fp - idsw) / max(1.0, gt_dets),
        "CLR_F1": tp / max(1.0, tp + 0.5 * fn + 0.5 * fp),
        "FP_per_frame": fp / max(1.0, frames),
        "MOTAL": motal,
    }


def _identity_expected(*, idtp: float, idfn: float, idfp: float) -> dict:
    return {
        "IDTP": idtp,
        "IDFN": idfn,
        "IDFP": idfp,
        "IDR": idtp / max(1.0, idtp + idfn),
        "IDP": idtp / max(1.0, idtp + idfp),
        "IDF1": idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn),
    }


def test_perfect_tracking(tmp_path) -> None:
    # Predictions identical to GT: every ratio is exactly 1, every error count 0.
    gt = _boxes(1, 0.0, 5) + _boxes(2, 500.0, 5)
    dataset = _dataset(GtSequence(name="seq", num_timesteps=5, tracks=tuple(gt)))
    result = _evaluate(dataset, {"seq": gt}, tmp_path)

    scores = result.per_sequence["seq"]
    assert_scores_exact(
        scores["HOTA"],
        _hota_expected(
            tp=10, fn=0, fp=0, det_a=1, det_re=1, det_pr=1, ass_a=1, ass_re=1, ass_pr=1, loca=1
        ),
    )
    assert_scores_exact(
        scores["CLEAR"],
        _clear_expected(tp=10, fn=0, fp=0, idsw=0, mt=2, pt=0, ml=0, frag=0, frames=5, motal=1.0),
    )
    assert_scores_exact(scores["Identity"], _identity_expected(idtp=10, idfn=0, idfp=0))
    assert_scores_exact(scores["Count"], {"Dets": 10.0, "GT_Dets": 10.0, "IDs": 2.0, "GT_IDs": 2.0})


def test_half_of_gt_missed(tmp_path) -> None:
    # 10 GT dets; only object 1 predicted (perfectly), object 2 entirely missed.
    # DetA = TP/(TP+FN+FP) = 5/10; AssA = 1 (the one pred track is pure), so
    # HOTA = sqrt(0.5) at every alpha. MOTA = (5-0-0)/10; IDF1 = 5/(5+0+2.5) = 2/3.
    # Object 1 is tracked 5/5 -> MT; object 2 0/5 -> ML.
    gt = _boxes(1, 0.0, 5) + _boxes(2, 500.0, 5)
    pred = _boxes(1, 0.0, 5)
    dataset = _dataset(GtSequence(name="seq", num_timesteps=5, tracks=tuple(gt)))
    result = _evaluate(dataset, {"seq": pred}, tmp_path)

    scores = result.per_sequence["seq"]
    assert_scores_exact(
        scores["HOTA"],
        _hota_expected(
            tp=5,
            fn=5,
            fp=0,
            det_a=0.5,
            det_re=0.5,
            det_pr=1.0,
            ass_a=1.0,
            ass_re=1.0,
            ass_pr=1.0,
            loca=1.0,
        ),
    )
    assert_scores_exact(
        scores["CLEAR"],
        _clear_expected(tp=5, fn=5, fp=0, idsw=0, mt=1, pt=0, ml=1, frag=0, frames=5, motal=0.5),
    )
    assert_scores_exact(scores["Identity"], _identity_expected(idtp=5, idfn=5, idfp=0))
    assert_scores_exact(scores["Count"], {"Dets": 5.0, "GT_Dets": 10.0, "IDs": 1.0, "GT_IDs": 2.0})


def test_exactly_one_id_switch(tmp_path) -> None:
    # One GT track over 4 frames; tracker id 1 covers frames 1-2, id 2 frames 3-4.
    # CLEAR: TP=4, one switch -> MOTA = (4-0-1)/4 = 0.75. MOTAL's log10(IDSW)
    # quirk makes a single switch free: log10(1) = 0 -> MOTAL = 1.0.
    # Identity can credit only one tracker id: IDTP=2, IDFN=IDFP=2 -> IDF1=0.5.
    # HOTA association: each pred track overlaps gt for 2 of 4 frames;
    # ass_a = 2/(4+2-2) = 0.5 per TP -> AssA=0.5, AssRe=0.5 (2/4), AssPr=1 (2/2),
    # DetA=1 -> HOTA = sqrt(0.5).
    gt = _boxes(1, 0.0, 4)
    pred = [Track(frame=f, track_id=1, x=0.0 + f, y=10.0, w=20.0, h=20.0, conf=1.0) for f in (1, 2)]
    pred += [
        Track(frame=f, track_id=2, x=0.0 + f, y=10.0, w=20.0, h=20.0, conf=1.0) for f in (3, 4)
    ]
    dataset = _dataset(GtSequence(name="seq", num_timesteps=4, tracks=tuple(gt)))
    result = _evaluate(dataset, {"seq": pred}, tmp_path)

    scores = result.per_sequence["seq"]
    assert_scores_exact(
        scores["HOTA"],
        _hota_expected(
            tp=4,
            fn=0,
            fp=0,
            det_a=1.0,
            det_re=1.0,
            det_pr=1.0,
            ass_a=0.5,
            ass_re=0.5,
            ass_pr=1.0,
            loca=1.0,
        ),
    )
    assert_scores_exact(
        scores["CLEAR"],
        _clear_expected(tp=4, fn=0, fp=0, idsw=1, mt=1, pt=0, ml=0, frag=0, frames=4, motal=1.0),
    )
    assert_scores_exact(scores["Identity"], _identity_expected(idtp=2, idfn=2, idfp=2))
    assert_scores_exact(scores["Count"], {"Dets": 4.0, "GT_Dets": 4.0, "IDs": 2.0, "GT_IDs": 1.0})


def test_conf_zero_gt_dropped_but_matched_pred_stays(tmp_path) -> None:
    # One frame. GT: id 1 (conf 1) at A, id 9 (conf 0, "do not consider") at B.
    # Preds: id 1 at A, id 2 at B. The protocol drops the conf-zero GT row
    # (GT_Dets=1, GT_IDs=1), but its matched prediction is NOT removed (only
    # distractor-class matches are) so it counts as a plain FP:
    # MOTA = (1-1-0)/1 = 0, CLR_Pr = 0.5, HOTA DetA = 1/2.
    gt = [
        Track(frame=1, track_id=1, x=0.0, y=10.0, w=20.0, h=20.0, conf=1.0),
        Track(frame=1, track_id=9, x=500.0, y=10.0, w=20.0, h=20.0, conf=0.0),
    ]
    pred = [
        Track(frame=1, track_id=1, x=0.0, y=10.0, w=20.0, h=20.0, conf=1.0),
        Track(frame=1, track_id=2, x=500.0, y=10.0, w=20.0, h=20.0, conf=1.0),
    ]
    dataset = _dataset(GtSequence(name="seq", num_timesteps=1, tracks=tuple(gt)))
    result = _evaluate(dataset, {"seq": pred}, tmp_path)

    scores = result.per_sequence["seq"]
    assert_scores_exact(scores["Count"], {"Dets": 2.0, "GT_Dets": 1.0, "IDs": 2.0, "GT_IDs": 1.0})
    assert_scores_exact(
        scores["CLEAR"],
        _clear_expected(tp=1, fn=0, fp=1, idsw=0, mt=1, pt=0, ml=0, frag=0, frames=1, motal=0.0),
    )
    assert_scores_exact(scores["Identity"], _identity_expected(idtp=1, idfn=0, idfp=1))
    assert_scores_exact(
        scores["HOTA"],
        _hota_expected(
            tp=1,
            fn=0,
            fp=1,
            det_a=0.5,
            det_re=1.0,
            det_pr=0.5,
            ass_a=1.0,
            ass_re=1.0,
            ass_pr=1.0,
            loca=1.0,
        ),
    )


def test_multi_sequence_combine_is_detection_weighted(tmp_path) -> None:
    # Sequence "perfect" is tracked perfectly; sequence "missed" has no preds.
    # Combining sums TP/FN/FP (TrackEval-style detection weighting), it does NOT
    # average the per-sequence scores: combined TP=5, FN=5 -> DetA=0.5, while
    # AssA is TP-weighted -> (1*5 + 0*0)/5 = 1, so combined HOTA = sqrt(0.5),
    # strictly between the per-sequence extremes 0 and 1 (and != the 0.5 mean).
    gt = _boxes(1, 0.0, 5)
    dataset = _dataset(
        GtSequence(name="perfect", num_timesteps=5, tracks=tuple(gt)),
        GtSequence(name="missed", num_timesteps=5, tracks=tuple(gt)),
    )
    result = _evaluate(dataset, {"perfect": gt}, tmp_path)  # no file for "missed"

    per_perfect = result.per_sequence["perfect"]
    per_missed = result.per_sequence["missed"]
    assert np.array_equal(per_perfect["HOTA"]["HOTA"], _alphas(1.0))
    assert np.array_equal(per_missed["HOTA"]["HOTA"], _alphas(0.0))

    combined = result.combined
    assert_scores_exact(
        combined["HOTA"],
        _hota_expected(
            tp=5,
            fn=5,
            fp=0,
            det_a=0.5,
            det_re=0.5,
            det_pr=1.0,
            ass_a=1.0,
            ass_re=1.0,
            ass_pr=1.0,
            loca=1.0,
        ),
    )
    assert np.all(_alphas(0.0) < combined["HOTA"]["HOTA"])
    assert np.all(combined["HOTA"]["HOTA"] < _alphas(1.0))
    assert combined["HOTA"]["HOTA"][0] == SQRT_HALF

    # CLR_Frames quirk: the empty-preds early return never sets CLR_Frames, so
    # "missed" contributes 0 frames and the combined count stays 5, not 10.
    assert_scores_exact(
        combined["CLEAR"],
        _clear_expected(tp=5, fn=5, fp=0, idsw=0, mt=1, pt=0, ml=1, frag=0, frames=5, motal=0.5),
    )
    assert_scores_exact(combined["Identity"], _identity_expected(idtp=5, idfn=5, idfp=0))
    assert_scores_exact(
        combined["Count"], {"Dets": 5.0, "GT_Dets": 10.0, "IDs": 1.0, "GT_IDs": 2.0}
    )


def test_empty_gt_sequence(tmp_path) -> None:
    # No GT at all; 5 predictions. Every prediction is a false positive.
    # CLEAR's empty-GT early return sets only CLR_FP and MLR=1.0 (upstream quirk:
    # MLR is 1 despite zero GT tracks) and skips the derived-field computation.
    pred = _boxes(1, 0.0, 5)
    dataset = _dataset(GtSequence(name="seq", num_timesteps=5, tracks=()))
    result = _evaluate(dataset, {"seq": pred}, tmp_path)

    scores = result.per_sequence["seq"]
    assert_scores_exact(scores["Count"], {"Dets": 5.0, "GT_Dets": 0.0, "IDs": 1.0, "GT_IDs": 0.0})
    hota_expected: dict[str, float | np.ndarray] = {
        field: _alphas(0.0) for field in scores["HOTA"] if field.count("(") == 0
    }
    hota_expected["HOTA_FP"] = _alphas(5.0)
    hota_expected["LocA"] = _alphas(1.0)
    hota_expected["HOTA(0)"] = 0.0
    hota_expected["LocA(0)"] = 1.0
    hota_expected["HOTALocA(0)"] = 0.0
    assert_scores_exact(scores["HOTA"], hota_expected)
    clear_expected = dict.fromkeys(scores["CLEAR"], 0.0)
    clear_expected["CLR_FP"] = 5.0
    clear_expected["MLR"] = 1.0
    assert_scores_exact(scores["CLEAR"], clear_expected)
    assert_scores_exact(scores["Identity"], _identity_expected(idtp=0, idfn=0, idfp=5))


def test_empty_predictions_sequence(tmp_path) -> None:
    # GT present (10 dets, 2 ids), no prediction file: everything is missed.
    # CLEAR's empty-preds early return sets CLR_FN, ML and MLR only.
    gt = _boxes(1, 0.0, 5) + _boxes(2, 500.0, 5)
    dataset = _dataset(GtSequence(name="seq", num_timesteps=5, tracks=tuple(gt)))
    result = _evaluate(dataset, {}, tmp_path)

    scores = result.per_sequence["seq"]
    assert_scores_exact(scores["Count"], {"Dets": 0.0, "GT_Dets": 10.0, "IDs": 0.0, "GT_IDs": 2.0})
    hota_expected: dict[str, float | np.ndarray] = {
        field: _alphas(0.0) for field in scores["HOTA"] if field.count("(") == 0
    }
    hota_expected["HOTA_FN"] = _alphas(10.0)
    hota_expected["LocA"] = _alphas(1.0)
    hota_expected["HOTA(0)"] = 0.0
    hota_expected["LocA(0)"] = 1.0
    hota_expected["HOTALocA(0)"] = 0.0
    assert_scores_exact(scores["HOTA"], hota_expected)
    clear_expected = dict.fromkeys(scores["CLEAR"], 0.0)
    clear_expected["CLR_FN"] = 10.0
    clear_expected["ML"] = 2.0
    clear_expected["MLR"] = 1.0
    assert_scores_exact(scores["CLEAR"], clear_expected)
    assert_scores_exact(scores["Identity"], _identity_expected(idtp=0, idfn=10, idfp=0))
