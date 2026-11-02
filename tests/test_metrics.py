import numpy as np
import pytest

from moteval import (
    CLEAR,
    HOTA,
    Count,
    Identity,
    JAndF,
    MOTDataset,
    Protocol,
    TrackMAP,
    evaluate,
)
from moteval.data.model import FrameConvention, GtSequence
from moteval.formats import Track, write_mot

GOLDEN_PROTOCOL = Protocol(
    name="golden",
    frame_convention=FrameConvention(name="1-indexed", first_frame=1),
    eval_classes=(1,),
)
METRICS = (HOTA, CLEAR, Identity, Count)


def _boxes(track_id: int, x0: float, frames: int, conf: float = 1.0) -> list[Track]:
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
    # TrackEval's MOTAL subtracts log10(IDSW), so one switch costs nothing: MOTAL = 1.0.
    assert_scores_exact(
        scores["CLEAR"],
        _clear_expected(tp=4, fn=0, fp=0, idsw=1, mt=1, pt=0, ml=0, frag=0, frames=4, motal=1.0),
    )
    assert_scores_exact(scores["Identity"], _identity_expected(idtp=2, idfn=2, idfp=2))
    assert_scores_exact(scores["Count"], {"Dets": 4.0, "GT_Dets": 4.0, "IDs": 2.0, "GT_IDs": 1.0})


def test_conf_zero_gt_dropped_but_matched_pred_stays(tmp_path) -> None:
    # TrackEval removes only predictions matched to distractor classes, so the prediction
    # on the zero-marked GT row stays and counts as a false positive.
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


def test_hota_alignment_ignores_overlap_at_or_below_eps(tmp_path) -> None:
    # TrackEval's HOTA alignment drops a similarity whose Jaccard denominator is at or below
    # eps: GT 2's IoU of ~7e-17 in frames 2 and 3 adds nothing, so frame 1 matches GT 1.
    box = {"y": 0.0, "w": 100.0, "h": 100.0, "conf": 1.0}
    gt = [Track(frame=f, track_id=1, x=0.0, **box) for f in (1, 4)]
    gt += [Track(frame=1, track_id=2, x=0.0, **box)]
    gt += [Track(frame=f, track_id=2, x=100.0 - 2.0**-46, **box) for f in (2, 3)]
    pred = [Track(frame=f, track_id=1, x=0.0, **box) for f in (1, 2, 3, 4)]
    dataset = _dataset(GtSequence(name="seq", num_timesteps=4, tracks=tuple(gt)))
    result = _evaluate(dataset, {"seq": pred}, tmp_path)

    assert_scores_exact(
        result.per_sequence["seq"]["HOTA"],
        _hota_expected(
            tp=2,
            fn=3,
            fp=2,
            det_a=2 / 7,
            det_re=2 / 5,
            det_pr=0.5,
            ass_a=0.5,
            ass_re=1.0,
            ass_pr=0.5,
            loca=1.0,
        ),
    )


def test_hota_alpha_threshold_is_eps_guarded(tmp_path) -> None:
    # IoU is exactly 0.75 and the 15th alpha, np.arange(0.05, 0.99, 0.05)[14], is
    # 0.7500000000000001: TrackEval's HOTA matches at `similarity >= alpha - eps`.
    gt = [Track(frame=1, track_id=1, x=0.0, y=0.0, w=7.0, h=10.0, conf=1.0)]
    pred = [Track(frame=1, track_id=1, x=1.0, y=0.0, w=7.0, h=10.0, conf=1.0)]
    dataset = _dataset(GtSequence(name="seq", num_timesteps=1, tracks=tuple(gt)))
    scores = _evaluate(dataset, {"seq": pred}, tmp_path).per_sequence["seq"]["HOTA"]

    matched = np.array([1.0] * 15 + [0.0] * 4)
    assert np.array_equal(scores["HOTA_TP"], matched)
    assert np.array_equal(scores["HOTA_FN"], 1.0 - matched)
    assert np.array_equal(scores["HOTA_FP"], 1.0 - matched)


def test_multi_sequence_combine_is_detection_weighted(tmp_path) -> None:
    gt = _boxes(1, 0.0, 5)
    dataset = _dataset(
        GtSequence(name="perfect", num_timesteps=5, tracks=tuple(gt)),
        GtSequence(name="missed", num_timesteps=5, tracks=tuple(gt)),
    )
    result = _evaluate(dataset, {"perfect": gt, "missed": []}, tmp_path)

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

    # TrackEval's CLEAR returns early on empty predictions before it sets CLR_Frames, so
    # "missed" adds 0 frames.
    assert_scores_exact(
        combined["CLEAR"],
        _clear_expected(tp=5, fn=5, fp=0, idsw=0, mt=1, pt=0, ml=1, frag=0, frames=5, motal=0.5),
    )
    assert_scores_exact(combined["Identity"], _identity_expected(idtp=5, idfn=5, idfp=0))
    assert_scores_exact(
        combined["Count"], {"Dets": 5.0, "GT_Dets": 10.0, "IDs": 1.0, "GT_IDs": 2.0}
    )


def test_empty_gt_sequence(tmp_path) -> None:
    # TrackEval's CLEAR returns early on empty GT with only CLR_FP and MLR = 1.0 set.
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
    # TrackEval's CLEAR returns early on empty predictions with only CLR_FN, ML and MLR set.
    gt = _boxes(1, 0.0, 5) + _boxes(2, 500.0, 5)
    dataset = _dataset(GtSequence(name="seq", num_timesteps=5, tracks=tuple(gt)))
    result = _evaluate(dataset, {"seq": []}, tmp_path)

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


FIRST_FIELD_READ = {
    HOTA: ("HOTA_TP", "HOTA_TP", "HOTA_TP"),
    CLEAR: ("CLR_TP", "CLR_TP", "CLR_TP"),
    Identity: ("IDTP", "IDTP", "IDTP"),
    Count: ("Dets", "Dets", "Dets"),
    TrackMAP: ("_dt_scores", "AP_all", "_num_dt_all"),
    JAndF: ("num_gt_tracks", "num_gt_tracks", "num_gt_tracks"),
}
COMBINERS = (
    "combine_sequences",
    "combine_classes_class_averaged",
    "combine_classes_det_averaged",
)


@pytest.mark.parametrize("metric", list(FIRST_FIELD_READ), ids=lambda m: m.__name__)
def test_combiner_input_missing_a_field_raises_key_error(metric) -> None:
    for method, field in zip(COMBINERS, FIRST_FIELD_READ[metric], strict=True):
        with pytest.raises(KeyError) as info:
            getattr(metric(), method)({"seq": {}})
        assert info.value.args == (field,), method
