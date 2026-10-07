import dataclasses
from dataclasses import replace

import numpy as np
import pytest

from moteval import CLEAR, HOTA, Identity, JAndF, TrackMAP, evaluate, load_dataset
from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol, RawFrame, preprocess_frame
from moteval.data.similarity import box_ioa, box_iou
from moteval.formats import Track, read_mot, write_mot
from moteval.metrics.count import Count
from tests.conftest import (
    TWO_CLASS_SEQUENCE,
    load_toy,
    load_two_class,
    write_perfect_predictions,
    write_two_class_predictions,
)
from tests.scenarios import (
    build_mots_dataset,
    build_mots_scenarios,
    predictions_dir,
    write_mots_scenario,
)

CONVENTION = FrameConvention("1-indexed", 1)
_PROTOCOL = Protocol(name="t", frame_convention=CONVENTION, eval_classes=(1,))

# --------------------------------------------------------------------- model


def _seq() -> GtSequence:
    # id 42 is listed first so first-seen order and sorted order disagree.
    tracks = (
        Track(frame=1, track_id=42, x=50, y=50, w=10, h=10, conf=1.0),
        Track(frame=1, track_id=7, x=0, y=0, w=10, h=10, conf=1.0),
        Track(frame=2, track_id=7, x=1, y=0, w=10, h=10, conf=1.0),
    )
    return GtSequence(name="s", num_timesteps=2, tracks=tracks)


def test_sequence_data_is_frozen():
    data = build_sequence_data(_seq(), (), _PROTOCOL, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(data, "name", "mutated")  # noqa: B010  (assignment form fails static checks)


def test_ids_are_densified_via_sorted_mapping():
    # raw ids 7 and 42 densify to 0 and 1, never a max-id-sized dense array.
    data = build_sequence_data(_seq(), (), _PROTOCOL, 1)
    assert data.num_gt_ids == 2
    np.testing.assert_array_equal(np.sort(data.gt_ids[0]), [0, 1])
    np.testing.assert_array_equal(data.gt_ids[1], [0])


# ----------------------------------------------------------------- protocol


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

    # each view keeps only its own class: class 1 sits at x=0, class 2 at x=100.
    for view, x in ((cls1, 0.0), (cls2, 100.0)):
        assert view.num_gt_ids == 1
        assert view.num_pred_ids == 1
        assert view.num_gt_dets == 2
        assert view.num_pred_dets == 2
        for t in range(2):
            np.testing.assert_array_equal(view.gt_boxes[t][:, 0], [x])
            np.testing.assert_array_equal(view.pred_boxes[t][:, 0], [x])


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


# -------------------------------------------------------------- similarity


def test_box_iou_against_hand_computed():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10]], dtype=np.float64)
    ious = box_iou(a, b)
    # identical boxes -> 1.0; half-overlap -> inter 50 / union 150 = 1/3.
    np.testing.assert_allclose(ious, [[1.0, 1 / 3], [1.0, 1 / 3]])


def test_box_iou_disjoint_is_zero():
    # disjoint along one axis each: an unclamped overlap would be (-90) * 10 < 0.
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[100, 0, 10, 10], [0, 100, 10, 10]], dtype=np.float64)
    np.testing.assert_array_equal(box_iou(a, b), [[0.0, 0.0]])


def test_box_iou_empty_sides():
    empty = np.zeros((0, 4), dtype=np.float64)
    full = np.array([[0, 0, 10, 10]], dtype=np.float64)
    assert box_iou(empty, full).shape == (0, 1)
    assert box_iou(full, empty).shape == (1, 0)


def test_box_ioa_against_hand_computed():
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[5, 0, 10, 10], [0, 0, 20, 20]], dtype=np.float64)
    ioas = box_ioa(a, b)
    # normalised by a's area (100): half-overlap -> 50/100; a fully inside b -> 100/100
    # (IoU there would be 0.25 -- IoA is asymmetric by design).
    np.testing.assert_allclose(ioas, [[0.5, 1.0]])


def test_box_ioa_zero_area_a_is_zero():
    a = np.array([[0, 0, 0, 0]], dtype=np.float64)
    b = np.array([[0, 0, 10, 10]], dtype=np.float64)
    assert box_ioa(a, b)[0, 0] == 0.0


def test_box_ioa_empty_sides():
    empty = np.zeros((0, 4), dtype=np.float64)
    full = np.array([[0, 0, 10, 10]], dtype=np.float64)
    assert box_ioa(empty, full).shape == (0, 1)
    assert box_ioa(full, empty).shape == (1, 0)


# ---------------------------------------------------------------- formats


def test_mot_txt_round_trip(tmp_path):
    tracks = [
        Track(frame=1, track_id=1, x=10.0, y=10.0, w=20.0, h=20.0, conf=1.0),
        Track(frame=2, track_id=1, x=12.0, y=10.0, w=20.0, h=20.0, conf=0.9),
        Track(frame=1, track_id=2, x=100.0, y=100.0, w=30.0, h=40.0, conf=0.5, class_id=3),
    ]
    path = tmp_path / "seq.txt"
    write_mot(path, tracks)

    expected = sorted(tracks, key=lambda t: (t.frame, t.track_id))
    assert read_mot(path, class_column=True) == expected


def test_read_mot_defaults_conf_when_absent(tmp_path):
    path = tmp_path / "seq.txt"
    path.write_text("1,1,10,10,20,20\n")
    (row,) = read_mot(path)
    assert row.conf == 1.0


def test_read_mot_malformed_numeric_row_names_file_and_line(tmp_path):
    path = tmp_path / "seq.txt"
    path.write_text("1,1,10,10,20,20\n2,2,x,10,20,20\n")
    with pytest.raises(ValueError) as exc:
        read_mot(path)
    message = str(exc.value)
    assert str(path) in message
    assert ":2" in message


def test_read_mot_reads_the_class_column_only_when_asked(tmp_path):
    path = tmp_path / "seq.txt"
    path.write_text("1,1,10,10,20,20,1,3\n1,2,10,10,20,20,1,-1\n")
    assert [t.class_id for t in read_mot(path)] == [1, 1]
    path.write_text("1,1,10,10,20,20,1,3\n")
    assert [t.class_id for t in read_mot(path, class_column=True)] == [3]


# ------------------------------------------------------------ frame indexing
#
# Regression proof for the frame-indexing contract (issue #4). Historically
# track-zoo silently dropped frame 0 of 0-indexed predictions evaluated under
# a 1-indexed assumption (bit ChimpACT). These tests prove that failure mode
# is structurally impossible: out-of-range frames raise loudly through the real
# `evaluate()` path, and a correctly declared 0-indexed benchmark evaluates
# identically to the same data re-encoded 1-indexed. Every prediction file
# below is numbered independently of the ground truth -- never derived from
# GT frame lists.


def test_zero_indexed_predictions_against_one_indexed_benchmark_raise_on_frame_0(tmp_path):
    dataset = load_toy()
    seq = dataset.sequences[0]
    pred_tracks = [Track(frame=f, track_id=999, x=0, y=0, w=10, h=10, conf=1.0) for f in range(5)]
    write_mot(tmp_path / f"{seq.name}.txt", pred_tracks)
    write_mot(tmp_path / f"{dataset.sequences[1].name}.txt", [])

    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count()])

    message = str(exc.value)
    assert "0" in message
    assert "1-indexed" in message


def test_predictions_beyond_sequence_length_raise_loud_error(tmp_path):
    dataset = load_toy()
    seq = dataset.sequences[0]
    pred_tracks = [Track(frame=6, track_id=999, x=0, y=0, w=10, h=10, conf=1.0)]
    write_mot(tmp_path / f"{seq.name}.txt", pred_tracks)
    write_mot(tmp_path / f"{dataset.sequences[1].name}.txt", [])

    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count()])

    message = str(exc.value)
    assert "6" in message
    assert "1-indexed" in message


def _gt_sequence(first_frame: int) -> GtSequence:
    tracks = tuple(
        Track(frame=f, track_id=tid, x=10, y=10, w=20, h=20, conf=1.0)
        for tid in (1, 2)
        for f in range(first_frame, first_frame + 5)
    )
    return GtSequence(name="s", num_timesteps=5, tracks=tracks)


def _predictions_zero_indexed() -> list[Track]:
    return [
        Track(frame=f, track_id=tid, x=0, y=0, w=10, h=10, conf=1.0)
        for tid in (100, 200)
        for f in range(5)
    ]


def _predictions_one_indexed() -> list[Track]:
    return [
        Track(frame=f, track_id=tid, x=0, y=0, w=10, h=10, conf=1.0)
        for tid in (100, 200)
        for f in range(1, 6)
    ]


def test_zero_indexed_benchmark_bit_identical_to_one_indexed_reencoding(tmp_path):
    zero_protocol = Protocol(
        name="zero", frame_convention=FrameConvention("0-indexed", 0), eval_classes=(1,)
    )
    one_protocol = Protocol(
        name="one", frame_convention=FrameConvention("1-indexed", 1), eval_classes=(1,)
    )
    zero_dataset = MOTDataset(
        name="zero", split="val", sequences=(_gt_sequence(0),), protocol=zero_protocol
    )
    one_dataset = MOTDataset(
        name="one", split="val", sequences=(_gt_sequence(1),), protocol=one_protocol
    )

    zero_dir, one_dir = tmp_path / "zero", tmp_path / "one"
    write_mot(zero_dir / "s.txt", _predictions_zero_indexed())
    write_mot(one_dir / "s.txt", _predictions_one_indexed())

    zero_result = evaluate(zero_dataset, zero_dir, [Count()])
    one_result = evaluate(one_dataset, one_dir, [Count()])

    assert zero_result.combined == one_result.combined
    # Frame index 0 (0-indexed frame 0 / 1-indexed frame 1) is never silently
    # dropped: all 5 frames' 2 predictions each are counted on both sides.
    assert zero_result.combined["Count"]["Dets"] == 10.0


# --------------------------------------------------------------- evaluate()


def test_evaluate_returns_per_sequence_and_combined_count(tmp_path):
    dataset = load_toy()
    write_perfect_predictions(dataset, tmp_path)

    result = evaluate(dataset, tmp_path, [Count()])

    # Each toy sequence: 2 ids over 5 frames = 10 dets.
    for seq in dataset.sequences:
        scores = result.per_sequence[seq.name]["Count"]
        assert scores == {"Dets": 10.0, "GT_Dets": 10.0, "IDs": 2.0, "GT_IDs": 2.0}

    assert result.combined["Count"] == {
        "Dets": 20.0,
        "GT_Dets": 20.0,
        "IDs": 4.0,
        "GT_IDs": 4.0,
    }


def _toy_with_offset_id_predictions(tmp_dir):
    dataset = load_toy()
    for seq in dataset.sequences:
        preds = [replace(t, track_id=t.track_id + 100) for t in seq.tracks]
        write_mot(tmp_dir / f"{seq.name}.txt", preds)
    return dataset, tmp_dir


def _mots_scenario(tmp_dir):
    scenario = build_mots_scenarios()[0]
    write_mots_scenario(tmp_dir, scenario)
    return build_mots_dataset(tmp_dir, scenario), predictions_dir(tmp_dir)


@pytest.mark.parametrize(
    ("metric_cls", "build", "has_per_sequence_entry"),
    [
        (HOTA, _toy_with_offset_id_predictions, True),
        (CLEAR, _toy_with_offset_id_predictions, True),
        (Identity, _toy_with_offset_id_predictions, True),
        (Count, _toy_with_offset_id_predictions, True),
        # TrackMAP pools detections across sequences: it declares no per-sequence field.
        (TrackMAP, _toy_with_offset_id_predictions, False),
        (JAndF, _mots_scenario, True),
    ],
    ids=["HOTA", "CLEAR", "Identity", "Count", "TrackMAP", "JAndF"],
)
def test_evaluate_results_hold_only_declared_fields(
    metric_cls, build, has_per_sequence_entry, tmp_path
):
    dataset, pred_dir = build(tmp_path)
    name = metric_cls.__name__
    fields = set(metric_cls.fields)

    result = evaluate(dataset, pred_dir, [metric_cls()])

    assert set(result.per_sequence) == {seq.name for seq in dataset.sequences}
    for scores in result.per_sequence.values():
        assert (name in scores) == has_per_sequence_entry
        if has_per_sequence_entry:
            assert set(scores[name]) <= fields
    assert set(result.combined[name]) == fields


def test_evaluate_names_every_missing_prediction_file(tmp_path):
    dataset = load_toy()
    # No prediction files written at all: both sequences are named in one error.
    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count()])

    message = str(exc.value)
    assert str(tmp_path) in message
    assert "toy-0001" in message
    assert "toy-0002" in message


def test_evaluate_checks_prediction_files_before_scoring(tmp_path):
    dataset = load_toy()
    # toy-0001 has an out-of-range frame that scoring would reject; toy-0002 has no
    # file. The missing-file error must come first, before any sequence is scored.
    bad_frame = Track(frame=6, track_id=999, x=0, y=0, w=10, h=10, conf=1.0)
    write_mot(tmp_path / "toy-0001.txt", [bad_frame])

    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count()])

    message = str(exc.value)
    assert "has no <seq>.txt file" in message
    assert "toy-0002" in message
    assert "1-indexed" not in message


def test_evaluate_rejects_duplicate_metric_classes(tmp_path):
    dataset = load_toy()
    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count(), Count()])
    assert "Count" in str(exc.value)


def test_evaluate_scores_each_class_and_both_class_combinations(tmp_path):
    write_two_class_predictions(tmp_path)
    result = evaluate(load_two_class(), tmp_path, [HOTA(), Count()])

    # A multi-class run reports through per_class and the two class combinations.
    assert result.per_sequence == {}
    assert result.combined == {}
    assert list(result.per_class) == [1, 2]
    # Hand derivation. Every IoU is 1 or 0, so every HOTA alpha agrees.
    # Class 1: TP 2, FN 0, FP 1 (track 300 on a class-2 box) -> DetA 2/3.
    # Class 2: TP 2, FN 2 (GT 30), FP 0 -> DetA 1/2.
    class_1, class_2 = result.per_class[1], result.per_class[2]
    assert class_1.combined["Count"] == {"Dets": 3.0, "GT_Dets": 2.0, "IDs": 2.0, "GT_IDs": 1.0}
    assert class_2.combined["Count"] == {"Dets": 2.0, "GT_Dets": 4.0, "IDs": 1.0, "GT_IDs": 2.0}
    assert class_2.per_sequence[TWO_CLASS_SEQUENCE]["Count"] == class_2.combined["Count"]
    np.testing.assert_allclose(class_1.combined["HOTA"]["DetA"], 2 / 3)
    np.testing.assert_allclose(class_2.combined["HOTA"]["DetA"], 1 / 2)
    # Count sums over the classes in both combinations.
    totals = {"Dets": 5.0, "GT_Dets": 6.0, "IDs": 3.0, "GT_IDs": 3.0}
    assert result.class_averaged["Count"] == totals
    assert result.det_averaged["Count"] == totals
    # Class-averaged DetA is the mean of the class values: (2/3 + 1/2) / 2 = 7/12.
    # Det-averaged DetA pools the detections: 4 TP / (4 TP + 2 FN + 1 FP) = 4/7.
    np.testing.assert_allclose(result.class_averaged["HOTA"]["DetA"], 7 / 12)
    np.testing.assert_allclose(result.det_averaged["HOTA"]["DetA"], 4 / 7)


@pytest.mark.parametrize(
    ("row", "message"),
    [("1,100,0,0,10,10,1,-1", "without a class"), ("1,100,0,0,10,10,1", "malformed")],
    ids=["unset-class", "no-class-column"],
)
def test_multi_class_predictions_must_give_a_class(tmp_path, row, message):
    (tmp_path / f"{TWO_CLASS_SEQUENCE}.txt").write_text(row + "\n")
    with pytest.raises(ValueError, match=message):
        evaluate(load_two_class(), tmp_path, [Count()])


def test_trackmap_det_averaged_weights_classes_by_detection_tracks(tmp_path):
    # Every track is a 10x10 box over 2 frames, so all are in area_s and time_s.
    # Class 1: GT 1 and one matching prediction track 101.
    # Class 2: GT 2, a far-away false-positive track 201 with the higher score (0.9),
    # and a matching track 202 (score 0.5).
    tracks = (
        Track(frame=f, track_id=tid, x=x, y=x, w=10, h=10, conf=1.0, class_id=cls)
        for tid, x, cls in [(1, 0.0, 1), (2, 100.0, 2)]
        for f in (1, 2)
    )
    dataset = MOTDataset(
        name="trackmap-two-class",
        split="val",
        sequences=(GtSequence(name="s", num_timesteps=2, tracks=tuple(tracks)),),
        protocol=Protocol("p", CONVENTION, eval_classes=(1, 2)),
    )
    rows = [
        f"{f},{tid},{x},{x},10,10,{score},{cls}"
        for tid, x, score, cls in [(101, 0, 0.9, 1), (201, 500, 0.9, 2), (202, 100, 0.5, 2)]
        for f in (1, 2)
    ]
    (tmp_path / "s.txt").write_text("\n".join(rows) + "\n")

    result = evaluate(dataset, tmp_path, [TrackMAP()])

    # At every IoU threshold, class 1 has one true positive: precision 1 at every
    # recall threshold, so AP = 1 (up to TrackEval's np.spacing(1) in the precision
    # denominator). Class 2 ranks the false positive first: precision is 0 then 1/2,
    # which the right-to-left maximum makes 1/2 at every recall threshold, so AP = 1/2.
    # The detection weights are the non-ignored prediction tracks: 1 and 2.
    # Class-averaged AP = (1 + 1/2) / 2 = 3/4.
    # Det-averaged AP = (1 * 1 + 1/2 * 2) / (1 + 2) = 2/3.
    np.testing.assert_allclose(result.class_averaged["TrackMAP"]["AP_all"], np.full(10, 3 / 4))
    np.testing.assert_allclose(result.det_averaged["TrackMAP"]["AP_all"], np.full(10, 2 / 3))
    # The weights are private state: per-class results hold only the declared fields.
    for class_result in result.per_class.values():
        assert set(class_result.combined["TrackMAP"]) == set(TrackMAP.fields)


# ---------------------------------------------------------- extensibility


def test_custom_dataset_evaluates_through_public_api(tmp_path):
    import json

    import moteval

    annotations_path = tmp_path / "custom-annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {"frame": 1, "id": 1, "box": [0, 0, 10, 10], "class_id": 1},
                {"frame": 1, "id": 2, "box": [500, 500, 10, 10], "class_id": 2},
            ]
        )
    )
    convention = moteval.FrameConvention(name="test-ext-1-indexed", first_frame=1)
    distractor_protocol = moteval.Protocol(
        name="test-ext-distractors",
        frame_convention=convention,
        eval_classes=(1,),
        distractor_classes=(2,),
    )

    def load_custom_dataset() -> moteval.MOTDataset:
        rows = json.loads(annotations_path.read_text())
        tracks = tuple(
            moteval.Track(
                frame=row["frame"],
                track_id=row["id"],
                x=row["box"][0],
                y=row["box"][1],
                w=row["box"][2],
                h=row["box"][3],
                conf=1.0,
                class_id=row["class_id"],
            )
            for row in rows
        )
        sequence = moteval.GtSequence(name="custom-sequence", num_timesteps=1, tracks=tracks)
        return moteval.MOTDataset(
            name="test-ext-json-distractors",
            split="test",
            sequences=(sequence,),
            protocol=distractor_protocol,
        )

    dataset = load_custom_dataset()
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "custom-sequence.txt").write_text(
        "1,101,0,0,10,10,1\n1,202,500,500,10,10,1\n"
    )

    with_distractor = moteval.evaluate(
        dataset,
        predictions_dir,
        [moteval.HOTA(), moteval.CLEAR(), moteval.Identity(), moteval.Count()],
    ).combined

    assert with_distractor["HOTA"]["HOTA(0)"] == 1.0
    assert with_distractor["CLEAR"]["MOTA"] == 1.0
    assert with_distractor["CLEAR"]["CLR_FP"] == 0.0
    assert with_distractor["Identity"] == {
        "IDF1": 1.0,
        "IDR": 1.0,
        "IDP": 1.0,
        "IDTP": 1.0,
        "IDFN": 0.0,
        "IDFP": 0.0,
    }
    assert with_distractor["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 1.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }

    no_distractor_dataset = moteval.MOTDataset(
        name=dataset.name,
        split=dataset.split,
        sequences=dataset.sequences,
        protocol=moteval.Protocol(
            name="test-ext-no-distractors",
            frame_convention=convention,
            eval_classes=(1,),
        ),
    )
    without_distractor = moteval.evaluate(
        no_distractor_dataset,
        predictions_dir,
        [moteval.HOTA(), moteval.CLEAR(), moteval.Identity(), moteval.Count()],
    ).combined

    assert without_distractor["HOTA"]["HOTA(0)"] == 2**-0.5
    assert without_distractor["CLEAR"]["MOTA"] == 0.0
    assert without_distractor["CLEAR"]["CLR_FP"] == 1.0
    assert without_distractor["Identity"]["IDFP"] == 1.0
    assert without_distractor["Identity"]["IDF1"] == 2 / 3
    assert without_distractor["Count"] == {
        "Dets": 2.0,
        "GT_Dets": 1.0,
        "IDs": 2.0,
        "GT_IDs": 1.0,
    }


def test_unknown_dataset_lists_available_names():
    with pytest.raises(ValueError) as exc_info:
        load_dataset("nonexistent")

    message = str(exc_info.value)
    assert "available:" in message
    assert "dancetrack" in message
