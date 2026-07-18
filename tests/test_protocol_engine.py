import numpy as np

from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention, GtSequence
from moteval.data.protocol import Protocol, RawFrame, preprocess_frame
from moteval.formats.mot_txt import Track

CONVENTION = FrameConvention("1-indexed", 1)


def _raw(gt_classes, gt_conf, similarity, pred_classes, ignore_ioa=None):
    num_gt = len(gt_classes)
    num_pred = len(pred_classes)
    if ignore_ioa is None:
        ignore_ioa = np.zeros((num_pred, 0))
    return RawFrame(
        gt_ids=np.arange(num_gt, dtype=np.int64),
        gt_classes=np.array(gt_classes, dtype=np.int64),
        gt_conf=np.array(gt_conf, dtype=np.float64),
        gt_dets=np.zeros((num_gt, 4)),
        pred_ids=np.arange(100, 100 + num_pred, dtype=np.int64),
        pred_classes=np.array(pred_classes, dtype=np.int64),
        pred_confidences=np.ones(num_pred),
        pred_dets=np.zeros((num_pred, 4)),
        similarity=np.array(similarity, dtype=np.float64).reshape(num_gt, num_pred),
        ignore_ioa=np.array(ignore_ioa, dtype=np.float64).reshape(num_pred, -1),
    )


def test_prediction_matched_to_distractor_is_dropped_unmatched_survives():
    # gt0 pedestrian, gt1 distractor. pred0 -> gt0, pred1 -> gt1, pred2 unmatched.
    frame = _raw(
        gt_classes=[1, 99],
        gt_conf=[1, 1],
        similarity=[[0.9, 0.0, 0.0], [0.0, 0.9, 0.0]],
        pred_classes=[1, 1, 1],
    )
    protocol = Protocol("p", CONVENTION, eval_classes=(1,), distractor_classes=(99,))
    out = preprocess_frame(frame, protocol, 1)
    # pred1 (matched to distractor gt1) dropped; pred0 (pedestrian) and pred2 survive.
    np.testing.assert_array_equal(out.pred_ids, [100, 102])
    # only the pedestrian gt is kept for evaluation.
    assert out.gt_ids.shape[0] == 1


def test_unmatched_prediction_dropped_only_above_ignore_threshold():
    # no gt, three unmatched preds; IoA 0.6 drops, 0.5 (== threshold) and 0.4 survive.
    frame = _raw(
        gt_classes=[],
        gt_conf=[],
        similarity=np.zeros((0, 3)),
        pred_classes=[1, 1, 1],
        ignore_ioa=[[0.6], [0.5], [0.4]],
    )
    protocol = Protocol("p", CONVENTION, eval_classes=(1,), ignore_iou_threshold=0.5)
    out = preprocess_frame(frame, protocol, 1)
    np.testing.assert_array_equal(out.pred_ids, [101, 102])


def test_conf_zero_gt_still_matches_before_removal():
    # gt0 is a conf-zero distractor: it must participate in matching (removing the
    # matched prediction) before being dropped -- TrackEval's matching-then-removal.
    frame = _raw(
        gt_classes=[99],
        gt_conf=[0],
        similarity=[[0.9]],
        pred_classes=[1],
    )
    protocol = Protocol("p", CONVENTION, eval_classes=(1,), distractor_classes=(99,))
    out = preprocess_frame(frame, protocol, 1)
    assert out.pred_ids.shape[0] == 0
    assert out.gt_ids.shape[0] == 0


def test_conf_zero_gt_excluded_from_evaluation():
    # gt0 conf=1, gt1 conf=0 (both pedestrian); both preds matched, none distractor.
    frame = _raw(
        gt_classes=[1, 1],
        gt_conf=[1, 0],
        similarity=[[0.9, 0.0], [0.0, 0.9]],
        pred_classes=[1, 1],
    )
    protocol = Protocol("p", CONVENTION, eval_classes=(1,))
    out = preprocess_frame(frame, protocol, 1)
    # both predictions survive; the conf-zero gt row is excluded from evaluation.
    np.testing.assert_array_equal(out.pred_ids, [100, 101])
    np.testing.assert_array_equal(out.gt_ids, [0])


def _two_class_sequence():
    tracks = tuple(
        Track(frame=f, track_id=tid, x=x, y=x, w=10, h=10, conf=1.0, class_id=cls)
        for tid, x, cls in [(10, 0.0, 1), (20, 100.0, 2)]
        for f in (1, 2)
    )
    return GtSequence(name="s", num_timesteps=2, tracks=tracks)


def _two_class_predictions():
    return tuple(
        Track(frame=f, track_id=tid, x=x, y=x, w=10, h=10, conf=1.0, class_id=cls)
        for tid, x, cls in [(100, 0.0, 1), (200, 100.0, 2)]
        for f in (1, 2)
    )


def test_class_filtering_yields_per_class_views():
    protocol = Protocol("p", CONVENTION, eval_classes=(1, 2))
    gt = _two_class_sequence()
    pred = _two_class_predictions()

    cls1 = build_sequence_data(gt, pred, protocol, 1)
    cls2 = build_sequence_data(gt, pred, protocol, 2)

    for view in (cls1, cls2):
        assert view.num_gt_ids == 1
        assert view.num_pred_ids == 1
        assert view.num_gt_dets == 2
        assert view.num_pred_dets == 2


def test_ignore_regions_wired_through_conversion():
    # end-to-end: GtSequence.ignore_regions -> _bin_frames -> box_ioa -> engine.
    gt_tracks = tuple(Track(frame=f, track_id=1, x=0, y=0, w=10, h=10, conf=1.0) for f in (1, 2))
    ignore = tuple(Track(frame=f, track_id=0, x=100, y=100, w=20, h=20, conf=0.0) for f in (1, 2))
    gt = GtSequence(name="s", num_timesteps=2, tracks=gt_tracks, ignore_regions=ignore)
    preds = tuple(
        Track(frame=f, track_id=tid, x=x, y=x, w=10, h=10, conf=1.0)
        for tid, x in [(100, 0.0), (200, 105.0), (300, 500.0)]
        for f in (1, 2)
    )
    protocol = Protocol("p", CONVENTION, eval_classes=(1,))

    data = build_sequence_data(gt, preds, protocol, 1)

    # pred 200 sits fully inside the ignore region (IoA 1.0 > 0.5) while unmatched
    # and is dropped; matched pred 100 and far-away pred 300 survive.
    assert data.num_pred_ids == 2
    assert data.num_pred_dets == 4
    assert data.num_gt_ids == 1


def test_toy_protocol_passes_through_engine_unchanged():
    from moteval.benchmarks.toy import TOY_PROTOCOL, load_toy

    dataset = load_toy()
    seq = dataset.sequences[0]
    data = build_sequence_data(seq, seq.tracks, TOY_PROTOCOL, 1)
    # trivial protocol drops nothing: 2 ids over 5 frames on both sides.
    assert data.num_gt_ids == 2
    assert data.num_pred_ids == 2
    assert data.num_gt_dets == 10
    assert data.num_pred_dets == 10
