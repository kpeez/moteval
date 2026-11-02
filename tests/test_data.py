import configparser
import dataclasses
import json
from dataclasses import replace

import numpy as np
import pytest

from moteval import CLEAR, HOTA, Count, Identity, JAndF, TrackMAP, evaluate, load_dataset
from moteval._rust import python_error
from moteval.data.model import BoxGeometry, FrameConvention, GtSequence, MOTDataset, SequenceData
from moteval.data.protocol import Protocol
from moteval.formats import Track, read_mot, write_mot
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


def test_sequence_data_is_frozen():
    empty = (np.zeros(0),)
    data = SequenceData(
        name="s",
        num_timesteps=1,
        num_gt_ids=0,
        num_pred_ids=0,
        num_gt_dets=0,
        num_pred_dets=0,
        gt_ids=empty,
        pred_ids=empty,
        pred_confidences=empty,
        geometry=BoxGeometry(gt=(np.zeros((0, 4)),), pred=(np.zeros((0, 4)),)),
        similarity=(np.zeros((0, 0)),),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(data, "name", "mutated")  # noqa: B010  (assignment form fails static checks)


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
    assert zero_result.combined["Count"]["Dets"] == 10.0


def test_evaluate_returns_per_sequence_and_combined_count(tmp_path):
    dataset = load_toy()
    write_perfect_predictions(dataset, tmp_path)

    result = evaluate(dataset, tmp_path, [Count()])

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


@pytest.mark.parametrize(
    ("metric_cls", "build", "message"),
    [
        (
            TrackMAP,
            _mots_scenario,
            "TrackMAP reads boxes, but sequence 'SEQ-JF-01' of dataset 'parity-mots' holds masks",
        ),
        (
            JAndF,
            _toy_with_offset_id_predictions,
            "JAndF reads masks, but sequence 'toy-0001' of dataset 'toy' holds boxes",
        ),
    ],
    ids=["TrackMAP-on-masks", "JAndF-on-boxes"],
)
def test_evaluate_rejects_a_metric_that_reads_the_other_geometry(
    metric_cls, build, message, tmp_path
):
    dataset, pred_dir = build(tmp_path)
    with pytest.raises(ValueError, match=message):
        evaluate(dataset, pred_dir, [metric_cls()])


def test_evaluate_names_every_missing_prediction_file(tmp_path):
    dataset = load_toy()
    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count()])

    message = str(exc.value)
    assert str(tmp_path) in message
    assert "toy-0001" in message
    assert "toy-0002" in message


def test_evaluate_checks_prediction_files_before_scoring(tmp_path):
    dataset = load_toy()
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


def _toy_with_eval_classes(eval_classes: tuple[int, ...]) -> MOTDataset[GtSequence]:
    toy = load_toy()
    return replace(toy, protocol=replace(toy.protocol, eval_classes=eval_classes))


def test_evaluate_rejects_a_protocol_without_eval_classes(tmp_path):
    dataset = _toy_with_eval_classes(())
    write_perfect_predictions(dataset, tmp_path)
    with pytest.raises(ValueError, match="declares no eval_classes"):
        evaluate(dataset, tmp_path, [Count()])


def test_evaluate_rejects_a_dataset_without_sequences(tmp_path):
    dataset = replace(load_toy(), sequences=())
    with pytest.raises(ValueError, match="^dataset 'toy' has no sequences to score$"):
        evaluate(dataset, tmp_path, [HOTA(), CLEAR(), Identity(), Count(), TrackMAP()])


def test_evaluate_rejects_a_repeated_eval_class(tmp_path):
    dataset = _toy_with_eval_classes((1, 1))
    write_perfect_predictions(dataset, tmp_path)
    with pytest.raises(ValueError, match="repeats a class"):
        evaluate(dataset, tmp_path, [Count()])


def test_evaluate_scores_each_class_and_both_class_combinations(tmp_path):
    write_two_class_predictions(tmp_path)
    result = evaluate(load_two_class(), tmp_path, [HOTA(), Count()])

    assert result.per_sequence == {}
    assert result.combined == {}
    assert list(result.per_class) == [1, 2]
    class_1, class_2 = result.per_class[1], result.per_class[2]
    assert class_1.combined["Count"] == {"Dets": 3.0, "GT_Dets": 2.0, "IDs": 2.0, "GT_IDs": 1.0}
    assert class_2.combined["Count"] == {"Dets": 2.0, "GT_Dets": 4.0, "IDs": 1.0, "GT_IDs": 2.0}
    assert class_2.per_sequence[TWO_CLASS_SEQUENCE]["Count"] == class_2.combined["Count"]
    np.testing.assert_allclose(class_1.combined["HOTA"]["DetA"], 2 / 3)
    np.testing.assert_allclose(class_2.combined["HOTA"]["DetA"], 1 / 2)
    totals = {"Dets": 5.0, "GT_Dets": 6.0, "IDs": 3.0, "GT_IDs": 3.0}
    assert result.class_averaged["Count"] == totals
    assert result.det_averaged["Count"] == totals
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

    np.testing.assert_allclose(result.class_averaged["TrackMAP"]["AP_all"], np.full(10, 3 / 4))
    np.testing.assert_allclose(result.det_averaged["TrackMAP"]["AP_all"], np.full(10, 2 / 3))
    for class_result in result.per_class.values():
        assert set(class_result.combined["TrackMAP"]) == set(TrackMAP.fields)


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


def test_distractor_and_ignore_thresholds_are_eps_guarded(tmp_path):
    # TrackEval's eps guards: IoU 0.49999999999999994 still matches the distractor
    # (`>= 0.5 - eps`), and IoA 0.5000000000000001 is not an ignore drop (`> 0.5 + eps`).
    distractor = Track(frame=1, track_id=2, x=0.0, y=0.0, w=1.0, h=1.0, conf=1, class_id=2)
    ignore = Track(frame=1, track_id=-1, x=0.4999999999999999, y=300.0, w=5.0, h=1.0, conf=1)
    sequence = GtSequence(
        name="edge", num_timesteps=1, tracks=(distractor,), ignore_regions=(ignore,)
    )
    protocol = Protocol(
        name="t-eps", frame_convention=CONVENTION, eval_classes=(1,), distractor_classes=(2,)
    )
    dataset = MOTDataset(name="t", split="test", sequences=(sequence,), protocol=protocol)
    (tmp_path / "edge.txt").write_text("1,12,0.33333333333333337,0,1,1,0.8\n1,13,0,300,1,1,0.7\n")
    result = evaluate(dataset, tmp_path, [Count()])
    assert result.combined["Count"]["Dets"] == 1.0


def test_matching_fill_decides_which_prediction_a_distractor_removes(tmp_path):
    # TrackEval fills sub-0.5 cells with 0 on the box path and -10000 on MOTS. -10000 first
    # maximizes the number of pairs at 0.5 or above, so `x` matches distractor `c`.
    import moteval

    convention = moteval.FrameConvention(name="test-fill-1-indexed", first_frame=1)
    boxes = {"a": 50, "b": 80, "c": 20, "x": 50, "y": 80, "z": 110}
    tracks = tuple(
        moteval.Track(
            frame=1, track_id=track_id, x=boxes[name], y=0, w=100, h=10, conf=1, class_id=cls
        )
        for track_id, name, cls in ((1, "a", 1), (2, "b", 1), (3, "c", 2))
    )
    sequence = moteval.GtSequence(name="fill-sequence", num_timesteps=1, tracks=tracks)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "fill-sequence.txt").write_text(
        "".join(f"1,{101 + i},{boxes[n]},0,100,10,1\n" for i, n in enumerate("xyz"))
    )

    def score(fill: float) -> dict:
        protocol = moteval.Protocol(
            name=f"test-fill-{fill}",
            frame_convention=convention,
            eval_classes=(1,),
            distractor_classes=(2,),
            matching_fill=fill,
        )
        dataset = moteval.MOTDataset(
            name="test-fill", split="test", sequences=(sequence,), protocol=protocol
        )
        return moteval.evaluate(dataset, predictions, [moteval.CLEAR(), moteval.Count()]).combined

    box_path = score(0.0)
    assert box_path["Count"]["Dets"] == 3.0
    assert (box_path["CLEAR"]["CLR_TP"], box_path["CLEAR"]["CLR_FP"]) == (2.0, 1.0)
    mots_path = score(-10000.0)
    assert mots_path["Count"]["Dets"] == 2.0
    assert (mots_path["CLEAR"]["CLR_TP"], mots_path["CLEAR"]["CLR_FP"]) == (2.0, 0.0)


def test_unknown_dataset_lists_available_names():
    with pytest.raises(ValueError) as exc_info:
        load_dataset("nonexistent")

    message = str(exc_info.value)
    assert "available:" in message
    assert "dancetrack" in message


def test_a_missing_file_raises_oserror_with_python_text(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(OSError) as exc:
        read_mot(missing)
    assert str(exc.value) == f"[Errno 2] No such file or directory: '{missing}'"


@pytest.mark.parametrize(
    ("exception", "message", "cls"),
    [
        ("ValueError", "bad row", ValueError),
        ("Unsupported", "too large", ValueError),
        ("UnicodeDecodeError", "'utf-8' codec can't decode", ValueError),
        ("OSError", "[Errno 21] Is a directory: '/x'", OSError),
        ("TypeError", "not a mapping", TypeError),
        ("IndexError", "boolean index did not match", IndexError),
        ("KeyError", "'bbox'", KeyError),
        ("KeyError", "17", KeyError),
        ("InterpolationError", "bad interpolation", configparser.InterpolationError),
    ],
)
def test_rust_errors_are_raised_as_their_python_class(exception, message, cls):
    err = python_error(exception, message)
    assert type(err) is cls
    assert str(err) == message


def test_rust_json_errors_keep_their_position():
    message = "Expecting property name enclosed in double quotes: line 2 column 3 (char 7)"
    err = python_error("JSONDecodeError", message)
    assert isinstance(err, json.JSONDecodeError)
    assert str(err) == message
    assert (err.msg, err.lineno, err.colno, err.pos) == (
        "Expecting property name enclosed in double quotes",
        2,
        3,
        7,
    )
