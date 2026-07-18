import json
from pathlib import Path

import pytest

from moteval import evaluate
from moteval.benchmarks.panaf500 import PANAF500_PROTOCOL, _xyxy_to_xywh, load_panaf500
from moteval.formats.mot_txt import write_mot
from moteval.metrics.count import Count


def _write_ann(root: Path, split: str, video_id: str, annotations: list[dict]) -> None:
    ann_dir = root / "annotations" / split
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{video_id}.json").write_text(
        json.dumps({"video": video_id, "annotations": annotations})
    )


def test_xyxy_to_xywh_hand_computed():
    assert _xyxy_to_xywh([10, 20, 60, 120]) == (10, 20, 50, 100)
    assert _xyxy_to_xywh([0, 0, 1, 1]) == (0, 0, 1, 1)
    assert _xyxy_to_xywh([5.5, 6.5, 15.5, 26.5]) == (5.5, 6.5, 10.0, 20.0)


def test_gt_boxes_converted_from_xyxy_to_xywh(tmp_path):
    _write_ann(
        tmp_path,
        "validation",
        "vid1",
        [
            {
                "frame_id": 1,
                "detections": [{"bbox": [10, 20, 60, 120], "ape_id": 0}],
            },
            {"frame_id": 2, "detections": []},
            {
                "frame_id": 3,
                "detections": [{"bbox": [0, 0, 10, 10], "ape_id": 1}],
            },
        ],
    )

    dataset = load_panaf500(root=tmp_path, split="validation")

    (seq,) = dataset.sequences
    assert seq.name == "vid1"
    assert {t.track_id for t in seq.tracks} == {0, 1}
    first = next(t for t in seq.tracks if t.frame == 1)
    assert (first.x, first.y, first.w, first.h) == (10, 20, 50, 100)
    third = next(t for t in seq.tracks if t.frame == 3)
    assert (third.x, third.y, third.w, third.h) == (0, 0, 10, 10)
    assert seq.num_timesteps == 3


def test_sequence_discovery_sorted_by_video_id(tmp_path):
    for video_id in ("vid2", "vid0", "vid1"):
        _write_ann(tmp_path, "train", video_id, [])

    dataset = load_panaf500(root=tmp_path, split="train")

    assert [s.name for s in dataset.sequences] == ["vid0", "vid1", "vid2"]


def test_missing_split_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="split directory not found"):
        load_panaf500(root=tmp_path, split="validation")


def test_gt_class_id_matches_protocol_eval_class(tmp_path):
    # Regression for the class_id hazard (issue #11 comment): this loader
    # never touches read_mot, so every Track must be stamped explicitly.
    _write_ann(
        tmp_path,
        "validation",
        "vid1",
        [{"frame_id": 1, "detections": [{"bbox": [0, 0, 10, 10], "ape_id": 0}]}],
    )

    dataset = load_panaf500(root=tmp_path, split="validation")

    (seq,) = dataset.sequences
    assert all(track.class_id in PANAF500_PROTOCOL.eval_classes for track in seq.tracks)


def test_end_to_end_evaluate_with_independently_numbered_predictions(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()

    _write_ann(
        gt_root,
        "validation",
        "vid1",
        [
            {"frame_id": 1, "detections": [{"bbox": [10, 20, 60, 120], "ape_id": 0}]},
            {"frame_id": 2, "detections": [{"bbox": [12, 22, 62, 122], "ape_id": 0}]},
        ],
    )

    dataset = load_panaf500(root=gt_root, split="validation")
    (seq,) = dataset.sequences
    write_mot(pred_dir / f"{seq.name}.txt", list(seq.tracks))

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 2.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }
