import json
from pathlib import Path

import pytest

from moteval import evaluate
from moteval.benchmarks.chimpact import (
    _CLASS_ID,
    _TEST_CLIPS,
    _VAL_CLIPS,
    load_chimpact,
)
from moteval.data.convert import build_sequence_data
from moteval.formats.mot_txt import Track, write_mot
from moteval.metrics.count import Count


def _coco_image(image_id: int, block: int) -> dict:
    return {"id": image_id, "file_name": f"{block:06d}.jpg"}


def _coco_ann(ann_id: int, image_id: int, bbox_id: int, bbox: list[float]) -> dict:
    # `category_id` is real ChimpACT schema (always 0, the single "Chimpanzee"
    # category) and is deliberately never read by the loader -- see below.
    return {"id": ann_id, "image_id": image_id, "bbox_id": bbox_id, "bbox": bbox, "category_id": 0}


def _write_clip(root: Path, clip: str, labels: dict) -> None:
    label_dir = root / "ChimpACT_release_v1" / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{clip}.json").write_text(json.dumps(labels))


def test_interpolation_matches_hand_derived_fixture(tmp_path):
    # Three keyframes (blocks 0, 1, 2); block 2 exists only so block 1 is not
    # the clip's terminal keyframe, keeping frames 0-10 pure interpolation,
    # untouched by the trailing hold-tail (tested separately below). Track
    # moves by (100, 50) in x/y between blocks 0 and 1; values hand-computed:
    # at frame f (0 <= f <= 10), x = 100 * f/10, y = 50 * f/10.
    labels = {
        "images": [_coco_image(0, 0), _coco_image(1, 1), _coco_image(2, 2)],
        "annotations": [
            _coco_ann(0, 0, 5, [0.0, 0.0, 20.0, 30.0]),
            _coco_ann(1, 1, 5, [100.0, 50.0, 20.0, 30.0]),
            _coco_ann(2, 2, 5, [100.0, 50.0, 20.0, 30.0]),
        ],
    }
    _write_clip(tmp_path, "clip-a", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    by_frame = {t.frame: t for t in seq.tracks if t.frame <= 10}
    assert set(by_frame) == set(range(11))
    for frame in range(11):
        t = frame / 10
        row = by_frame[frame]
        assert row.x == pytest.approx(100.0 * t)
        assert row.y == pytest.approx(50.0 * t)
        assert row.w == pytest.approx(20.0)
        assert row.h == pytest.approx(30.0)
        assert row.track_id == 5
    # Fractional intermediate frames are the interesting hand-checked cases.
    assert (by_frame[3].x, by_frame[3].y) == pytest.approx((30.0, 15.0))
    assert (by_frame[7].x, by_frame[7].y) == pytest.approx((70.0, 35.0))
    assert seq.num_timesteps == 30


def test_bbox_id_23_is_dropped(tmp_path):
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [
            _coco_ann(0, 0, 22, [0.0, 0.0, 10.0, 10.0]),
            _coco_ann(1, 0, 23, [5.0, 5.0, 10.0, 10.0]),
        ],
    }
    _write_clip(tmp_path, "clip-b", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert {t.track_id for t in seq.tracks} == {22}


def test_frame_zero_contributes(tmp_path):
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [_coco_ann(0, 0, 1, [1.0, 2.0, 3.0, 4.0])],
    }
    _write_clip(tmp_path, "clip-c", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert 0 in {t.frame for t in seq.tracks}
    frame_zero = next(t for t in seq.tracks if t.frame == 0)
    assert (frame_zero.x, frame_zero.y, frame_zero.w, frame_zero.h) == (1.0, 2.0, 3.0, 4.0)

    data = build_sequence_data(seq, seq.tracks, dataset.protocol, _CLASS_ID)
    # Frame 0 is the first timestep -- the silent-drop bug this rewrite
    # exists to kill would have discarded it.
    assert data.gt_ids[0].shape[0] == 1


def test_hold_last_box_when_track_has_no_match_in_next_keyframe(tmp_path):
    # Ported from legacy track-zoo's test_missing_next_keyframe_holds_last_box
    # (issue #12 decision: replicate legacy hold semantics exactly, for
    # comparability with published baselines). A single keyframe block for
    # the whole clip holds its box constant through all 9 interior frames
    # that follow it -- there is no next block to interpolate toward.
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [_coco_ann(0, 0, 0, [5.0, 5.0, 10.0, 10.0])],
    }
    _write_clip(tmp_path, "clip-hold", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert seq.num_timesteps == 10
    assert {t.frame for t in seq.tracks} == set(range(10))
    assert all((t.x, t.y, t.w, t.h) == (5.0, 5.0, 10.0, 10.0) for t in seq.tracks)
    assert all(t.track_id == 0 for t in seq.tracks)


def test_birth_death_boundaries(tmp_path):
    # Track 5 lives at blocks 0 and 1, then dies (absent from block 2, which
    # exists with a different track). Legacy holds a dead track's box for
    # exactly the one block of interior frames following its last keyframe,
    # then emits nothing further -- it does not retroactively reappear once
    # an unrelated later block exists. Track 7 is born at block 1 (no rows
    # before its first keyframe), survives into block 2 (interpolates), and
    # -- block 2 being the clip's terminal keyframe -- holds through the
    # clip's derived tail too.
    labels = {
        "images": [_coco_image(0, 0), _coco_image(1, 1), _coco_image(2, 2)],
        "annotations": [
            _coco_ann(0, 0, 5, [0.0, 0.0, 10.0, 10.0]),
            _coco_ann(1, 1, 5, [10.0, 10.0, 10.0, 10.0]),
            _coco_ann(2, 1, 7, [0.0, 0.0, 5.0, 5.0]),
            _coco_ann(3, 2, 7, [10.0, 10.0, 5.0, 5.0]),
        ],
    }
    _write_clip(tmp_path, "clip-d", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert seq.num_timesteps == 30

    track5_frames = sorted(t.frame for t in seq.tracks if t.track_id == 5)
    # Present at both block 0 (frame 0) and block 1 (frame 10): interior
    # frames 1-9 interpolate. Absent from block 2: holds through block 1's
    # own interior (frames 11-19, using block 1's box unchanged), then
    # nothing from frame 20 onward.
    assert track5_frames == list(range(0, 20))
    held = [t for t in seq.tracks if t.track_id == 5 and 11 <= t.frame <= 19]
    assert all((t.x, t.y, t.w, t.h) == (10.0, 10.0, 10.0, 10.0) for t in held)

    track7_frames = sorted(t.frame for t in seq.tracks if t.track_id == 7)
    # Born at block 1 (frame 10): no rows before it. Interpolates into block
    # 2 (frames 11-19), then holds through the clip's tail (frames 21-29)
    # since block 2 has no successor.
    assert track7_frames == list(range(10, 30))
    tail = [t for t in seq.tracks if t.track_id == 7 and 21 <= t.frame <= 29]
    assert all((t.x, t.y, t.w, t.h) == (10.0, 10.0, 5.0, 5.0) for t in tail)


def test_gt_class_id_matches_declared_constant_not_raw_coco_category(tmp_path):
    # Regression for the class_id hazard (issue #12 review comment): this
    # loader never touches read_mot, so every Track must be stamped
    # explicitly with the module's declared class rather than left to the
    # dataclass default. `_CLASS_ID` (1) happens to equal that default, so a
    # bare `class_id == 1` check would be tautological -- it can't tell an
    # intentional stamp from an omitted one. Give it teeth against a
    # different, real mistake instead: each fixture annotation also carries
    # the real ChimpACT COCO schema's `category_id` (always 0, "Chimpanzee"),
    # so a loader that blindly forwarded that raw field instead of stamping
    # its own explicit constant would produce `class_id == 0` here, not 1.
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [_coco_ann(0, 0, 1, [0.0, 0.0, 10.0, 10.0])],
    }
    _write_clip(tmp_path, "clip-e", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert all(track.class_id == _CLASS_ID for track in seq.tracks)
    assert all(track.class_id != 0 for track in seq.tracks)


def test_end_to_end_evaluate_with_independently_numbered_predictions(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()

    labels = {
        "images": [_coco_image(0, 0), _coco_image(1, 1)],
        "annotations": [
            _coco_ann(0, 0, 5, [10.0, 20.0, 60.0, 120.0]),
            _coco_ann(1, 1, 5, [12.0, 22.0, 60.0, 120.0]),
        ],
    }
    _write_clip(gt_root, "clip-f", labels)

    dataset = load_chimpact(root=gt_root, split="train")
    (seq,) = dataset.sequences
    # Predictions numbered independently of GT: different track id, and only
    # a disjoint one-frame subset of the twenty GT frames (frame 0, not the
    # rest of the interpolated/held span -- block 1 is this clip's terminal
    # keyframe, so it holds through frames 11-19 too).
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=0, track_id=900, x=11, y=21, w=60, h=120, conf=0.9, class_id=_CLASS_ID)],
    )

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 20.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }


def test_malformed_json_raises(tmp_path):
    label_dir = tmp_path / "ChimpACT_release_v1" / "labels"
    label_dir.mkdir(parents=True)
    (label_dir / "clip-g.json").write_text("{not valid json")

    with pytest.raises(json.JSONDecodeError):
        load_chimpact(root=tmp_path, split="train")


def test_missing_labels_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="labels directory not found"):
        load_chimpact(root=tmp_path, split="train")


def test_unknown_split_raises(tmp_path):
    _write_clip(
        tmp_path,
        "clip-h",
        {"images": [_coco_image(0, 0)], "annotations": [_coco_ann(0, 0, 1, [0, 0, 1, 1])]},
    )
    with pytest.raises(ValueError, match="unknown chimpact split"):
        load_chimpact(root=tmp_path, split="nope")


def test_split_partition_uses_official_clip_names(tmp_path):
    val_clip = _VAL_CLIPS[0]
    test_clip = _TEST_CLIPS[0]
    train_clip = "not-an-official-clip-name"
    tiny_labels = {"images": [_coco_image(0, 0)], "annotations": [_coco_ann(0, 0, 1, [0, 0, 1, 1])]}
    for clip in (val_clip, test_clip, train_clip):
        _write_clip(tmp_path, clip, tiny_labels)

    assert [s.name for s in load_chimpact(root=tmp_path, split="val").sequences] == [val_clip]
    assert [s.name for s in load_chimpact(root=tmp_path, split="test").sequences] == [test_clip]
    assert [s.name for s in load_chimpact(root=tmp_path, split="train").sequences] == [train_clip]
