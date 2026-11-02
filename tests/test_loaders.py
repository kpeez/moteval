import json
from pathlib import Path

import pytest

from moteval import CLEAR, Count, evaluate, load_dataset, load_motchallenge
from moteval.benchmarks import BENCHMARKS
from moteval.formats import Track, write_mot

load_animaltrack = BENCHMARKS["animaltrack"]
load_bft = BENCHMARKS["bft"]
load_chimpact = BENCHMARKS["chimpact"]
load_dancetrack = BENCHMARKS["dancetrack"]
load_gmot40 = BENCHMARKS["gmot40"]
load_panaf500 = BENCHMARKS["panaf500"]
load_sportsmot = BENCHMARKS["sportsmot"]
load_uavdt = BENCHMARKS["uavdt"]

_CHIMPACT_CLASS_ID = 1
_CHIMPACT_VAL_CLIP = "Azibo_ObsChimp_2015_11_25_d_clip_23000_24000"
_CHIMPACT_TEST_CLIP = "Azibo_ObsChimp_2015_11_25_d_clip_1000_2000"


def _write_bft_gt(root: Path, seq: str, rows: list[str]) -> None:
    ann_dir = root / "annotations_mot" / "val"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{seq}.txt").write_text("\n".join(rows) + "\n")


def _write_animaltrack_gt(root: Path, seq: str, rows: list[str]) -> None:
    gt_dir = root / "gt_all"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{seq}_gt.txt").write_text("\n".join(rows) + "\n")


def _write_gmot40_gt(root: Path, seq: str, rows: list[str]) -> None:
    gt_dir = root / "track_label"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{seq}.txt").write_text("\n".join(rows) + "\n")


def _load_bft(root: Path, split: str) -> object:
    return load_bft(root=root, split=split)


def _load_animaltrack(root: Path, split: str) -> object:
    return load_animaltrack(root=root, split="all")


def _load_gmot40(root: Path, split: str) -> object:
    return load_gmot40(root=root, split="test")


_MATRIX = (
    pytest.param(_load_bft, _write_bft_gt, "seqA", 0, id="bft"),
    pytest.param(_load_animaltrack, _write_animaltrack_gt, "chicken_1", 0, id="animaltrack"),
    pytest.param(_load_gmot40, _write_gmot40_gt, "bird-0", 1, id="gmot40"),
)


@pytest.mark.parametrize(("loader", "write_gt", "seq_name", "_offset"), _MATRIX)
def test_sequence_discovery(tmp_path, loader, write_gt, seq_name, _offset):
    write_gt(tmp_path, seq_name, ["1,1,10,10,20,20,1,1,1"])

    dataset = loader(tmp_path, "val")

    assert [s.name for s in dataset.sequences] == [seq_name]


@pytest.mark.parametrize(("loader", "write_gt", "seq_name", "offset"), _MATRIX)
def test_seq_length_derived_from_last_annotated_frame(tmp_path, loader, write_gt, seq_name, offset):
    rows = ["1,1,10,10,20,20,1,1,1", "3,1,12,10,20,20,1,1,1"]
    write_gt(tmp_path, seq_name, rows)

    dataset = loader(tmp_path, "val")

    (seq,) = dataset.sequences
    assert seq.num_timesteps == 3 + offset


@pytest.mark.parametrize(("loader", "write_gt", "seq_name", "_offset"), _MATRIX)
def test_end_to_end_evaluate_with_independently_numbered_predictions(
    tmp_path, loader, write_gt, seq_name, _offset
):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    write_gt(gt_root, seq_name, ["1,1,10,10,20,20,1,1,1", "2,1,12,10,20,20,1,1,1"])

    dataset = loader(gt_root, "val")
    (seq,) = dataset.sequences
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=1, track_id=901, x=11, y=11, w=20, h=20, conf=0.9)],
    )

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }


def test_animaltrack_train_test_selection_from_split_file(tmp_path):
    _write_animaltrack_gt(tmp_path, "chicken_1", ["1,1,10,10,20,20,1,1,1"])
    _write_animaltrack_gt(tmp_path, "deer_2", ["1,1,10,10,20,20,1,1,1"])
    split_dir = tmp_path / "train_test_splits"
    split_dir.mkdir()
    (split_dir / "videos_train.txt").write_text("chicken_1.mp4\n")
    (split_dir / "videos_test.txt").write_text("deer_2.mp4\n")

    train = load_animaltrack(root=tmp_path, split="train")
    test = load_animaltrack(root=tmp_path, split="test")

    assert [s.name for s in train.sequences] == ["chicken_1"]
    assert [s.name for s in test.sequences] == ["deer_2"]


def test_animaltrack_missing_split_file_raises(tmp_path):
    (tmp_path / "gt_all").mkdir()
    with pytest.raises(ValueError, match="not found at"):
        load_animaltrack(root=tmp_path, split="train")


def test_animaltrack_unknown_split_raises(tmp_path):
    (tmp_path / "gt_all").mkdir()
    with pytest.raises(ValueError, match="unknown animaltrack split"):
        load_animaltrack(root=tmp_path, split="nope")


def test_bft_missing_split_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="split directory not found"):
        load_bft(root=tmp_path, split="val")


def test_bft_empty_gt_raises_on_seq_length_derivation(tmp_path):
    _write_bft_gt(tmp_path, "seqA", [])
    with pytest.raises(ValueError, match="empty gt"):
        load_bft(root=tmp_path, split="val")


def test_bft_prediction_past_last_annotated_frame_raises(tmp_path):
    _write_bft_gt(tmp_path, "seqA", ["1,1,10,10,20,20,1,1,1"])
    dataset = load_bft(root=tmp_path, split="val")
    (seq,) = dataset.sequences
    assert seq.num_timesteps == 1

    pred_dir = tmp_path / "pred"
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=2, track_id=1, x=10, y=10, w=20, h=20, conf=0.9)],
    )
    with pytest.raises(ValueError, match="out of range"):
        evaluate(dataset, pred_dir, [Count()])


def test_gmot40_sequence_discovery_and_animal_subset(tmp_path):
    categories = ("airplane", "ball", "bird", "car", "fish", "insect", "person", "stock")
    animal_categories = {"bird", "fish", "insect", "stock"}
    for category in categories:
        _write_gmot40_gt(tmp_path, f"{category}-0", ["0,1,10,10,20,20,1,1,1"])

    dataset = load_gmot40(root=tmp_path, split="test")
    assert len(dataset.sequences) == len(categories)

    animal = load_gmot40(root=tmp_path, split="animal")
    assert {s.name for s in animal.sequences} == {f"{c}-0" for c in animal_categories}


def test_gmot40_unknown_split_raises(tmp_path):
    _write_gmot40_gt(tmp_path, "bird-0", ["0,1,10,10,20,20,1,1,1"])
    with pytest.raises(ValueError, match="unknown gmot40 split"):
        load_gmot40(root=tmp_path, split="nope")


def test_gmot40_frame_zero_contributes(tmp_path):
    _write_gmot40_gt(
        tmp_path,
        "bird-0",
        ["0,1,10,20,30,40,1,1,1", "1,1,12,22,30,40,1,1,1"],
    )

    dataset = load_gmot40(root=tmp_path, split="test")
    (seq,) = dataset.sequences

    assert sorted(t.frame for t in seq.tracks) == [0, 1]
    pred_dir = tmp_path / "pred"
    write_mot(pred_dir / "bird-0.txt", [Track(frame=0, track_id=5, x=10, y=20, w=30, h=40, conf=1)])
    clear = evaluate(dataset, pred_dir, [CLEAR()]).combined["CLEAR"]
    assert (clear["CLR_TP"], clear["CLR_FN"]) == (1.0, 1.0)


def _coco_image(image_id: int, block: int) -> dict:
    return {"id": image_id, "file_name": f"{block:06d}.jpg"}


def _coco_ann(ann_id: int, image_id: int, bbox_id: int, bbox: list[float]) -> dict:
    return {"id": ann_id, "image_id": image_id, "bbox_id": bbox_id, "bbox": bbox, "category_id": 0}


def _write_chimpact_clip(root: Path, clip: str, labels: dict) -> None:
    label_dir = root / "ChimpACT_release_v1" / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{clip}.json").write_text(json.dumps(labels))


def test_chimpact_frame_zero_contributes(tmp_path):
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [_coco_ann(0, 0, 1, [1.0, 2.0, 3.0, 4.0])],
    }
    _write_chimpact_clip(tmp_path, "clip-c", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences
    pred_dir = tmp_path / "pred"
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=0, track_id=5, x=1.0, y=2.0, w=3.0, h=4.0, conf=1.0)],
    )

    assert min(t.frame for t in seq.tracks) == 0
    assert evaluate(dataset, pred_dir, [CLEAR()]).combined["CLEAR"]["CLR_TP"] == 1.0


# ChimpACT's interpolation and hold rules copy the loader that produced the published
# baselines of arXiv:2511.02591 (issue #12).


def test_chimpact_interpolation_matches_hand_derived_fixture(tmp_path):
    labels = {
        "images": [_coco_image(0, 0), _coco_image(1, 1), _coco_image(2, 2)],
        "annotations": [
            _coco_ann(0, 0, 5, [0.0, 0.0, 20.0, 30.0]),
            _coco_ann(1, 1, 5, [100.0, 50.0, 20.0, 30.0]),
            _coco_ann(2, 2, 5, [100.0, 50.0, 20.0, 30.0]),
        ],
    }
    _write_chimpact_clip(tmp_path, "clip-a", labels)

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
    assert (by_frame[3].x, by_frame[3].y) == pytest.approx((30.0, 15.0))
    assert (by_frame[7].x, by_frame[7].y) == pytest.approx((70.0, 35.0))
    assert seq.num_timesteps == 30


def test_chimpact_bbox_id_23_is_dropped(tmp_path):
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [
            _coco_ann(0, 0, 22, [0.0, 0.0, 10.0, 10.0]),
            _coco_ann(1, 0, 23, [5.0, 5.0, 10.0, 10.0]),
        ],
    }
    _write_chimpact_clip(tmp_path, "clip-b", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert {t.track_id for t in seq.tracks} == {22}


def test_chimpact_hold_last_box_when_track_has_no_match_in_next_keyframe(tmp_path):
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [_coco_ann(0, 0, 0, [5.0, 5.0, 10.0, 10.0])],
    }
    _write_chimpact_clip(tmp_path, "clip-hold", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert seq.num_timesteps == 10
    assert {t.frame for t in seq.tracks} == set(range(10))
    assert all((t.x, t.y, t.w, t.h) == (5.0, 5.0, 10.0, 10.0) for t in seq.tracks)
    assert all(t.track_id == 0 for t in seq.tracks)


def test_chimpact_birth_death_boundaries(tmp_path):
    labels = {
        "images": [_coco_image(0, 0), _coco_image(1, 1), _coco_image(2, 2)],
        "annotations": [
            _coco_ann(0, 0, 5, [0.0, 0.0, 10.0, 10.0]),
            _coco_ann(1, 1, 5, [10.0, 10.0, 10.0, 10.0]),
            _coco_ann(2, 1, 7, [0.0, 0.0, 5.0, 5.0]),
            _coco_ann(3, 2, 7, [10.0, 10.0, 5.0, 5.0]),
        ],
    }
    _write_chimpact_clip(tmp_path, "clip-d", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert seq.num_timesteps == 30

    track5_frames = sorted(t.frame for t in seq.tracks if t.track_id == 5)
    assert track5_frames == list(range(20))
    held = [t for t in seq.tracks if t.track_id == 5 and 11 <= t.frame <= 19]
    assert all((t.x, t.y, t.w, t.h) == (10.0, 10.0, 10.0, 10.0) for t in held)

    track7_frames = sorted(t.frame for t in seq.tracks if t.track_id == 7)
    assert track7_frames == list(range(10, 30))
    tail = [t for t in seq.tracks if t.track_id == 7 and 21 <= t.frame <= 29]
    assert all((t.x, t.y, t.w, t.h) == (10.0, 10.0, 5.0, 5.0) for t in tail)


def test_chimpact_missing_intervening_keyframe_block_yields_zero_rows_for_that_window(tmp_path):
    labels = {
        "images": [_coco_image(0, 0), _coco_image(2, 2)],
        "annotations": [
            _coco_ann(0, 0, 5, [1.0, 1.0, 2.0, 2.0]),
            _coco_ann(1, 2, 9, [3.0, 3.0, 4.0, 4.0]),
        ],
    }
    _write_chimpact_clip(tmp_path, "clip-gap-block", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert seq.num_timesteps == 30
    assert {t.frame for t in seq.tracks}.isdisjoint(range(10, 20))

    track5 = [t for t in seq.tracks if t.track_id == 5]
    assert sorted(t.frame for t in track5) == list(range(10))
    assert all((t.x, t.y, t.w, t.h) == (1.0, 1.0, 2.0, 2.0) for t in track5)

    track9 = [t for t in seq.tracks if t.track_id == 9]
    assert sorted(t.frame for t in track9) == list(range(20, 30))
    assert all((t.x, t.y, t.w, t.h) == (3.0, 3.0, 4.0, 4.0) for t in track9)


def test_chimpact_multi_block_gap_then_reappearance_has_no_back_connection(tmp_path):
    labels = {
        "images": [_coco_image(0, 0), _coco_image(1, 1), _coco_image(2, 2), _coco_image(3, 3)],
        "annotations": [
            _coco_ann(0, 0, 5, [1.0, 1.0, 2.0, 2.0]),
            _coco_ann(1, 1, 9, [50.0, 50.0, 4.0, 4.0]),
            _coco_ann(2, 2, 9, [55.0, 55.0, 4.0, 4.0]),
            _coco_ann(3, 3, 5, [100.0, 100.0, 6.0, 6.0]),
        ],
    }
    _write_chimpact_clip(tmp_path, "clip-gap-track", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert seq.num_timesteps == 40

    track5_frames = sorted(t.frame for t in seq.tracks if t.track_id == 5)
    assert track5_frames == list(range(10)) + list(range(30, 40))
    before_gap = [t for t in seq.tracks if t.track_id == 5 and t.frame < 10]
    assert all((t.x, t.y, t.w, t.h) == (1.0, 1.0, 2.0, 2.0) for t in before_gap)
    after_gap = [t for t in seq.tracks if t.track_id == 5 and t.frame >= 30]
    assert all((t.x, t.y, t.w, t.h) == (100.0, 100.0, 6.0, 6.0) for t in after_gap)


def test_chimpact_gt_class_id_matches_declared_constant_not_raw_coco_category(tmp_path):
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [_coco_ann(0, 0, 1, [0.0, 0.0, 10.0, 10.0])],
    }
    _write_chimpact_clip(tmp_path, "clip-e", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert all(track.class_id == _CHIMPACT_CLASS_ID for track in seq.tracks)
    assert all(track.class_id != 0 for track in seq.tracks)


def test_chimpact_malformed_json_raises(tmp_path):
    label_dir = tmp_path / "ChimpACT_release_v1" / "labels"
    label_dir.mkdir(parents=True)
    (label_dir / "clip-g.json").write_text("{not valid json")

    with pytest.raises(json.JSONDecodeError):
        load_chimpact(root=tmp_path, split="train")


def test_chimpact_missing_labels_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="labels directory not found"):
        load_chimpact(root=tmp_path, split="train")


def test_chimpact_split_partition_uses_official_clip_names(tmp_path):
    val_clip = _CHIMPACT_VAL_CLIP
    test_clip = _CHIMPACT_TEST_CLIP
    train_clip = "not-an-official-clip-name"
    tiny_labels = {"images": [_coco_image(0, 0)], "annotations": [_coco_ann(0, 0, 1, [0, 0, 1, 1])]}
    for clip in (val_clip, test_clip, train_clip):
        _write_chimpact_clip(tmp_path, clip, tiny_labels)

    assert [s.name for s in load_chimpact(root=tmp_path, split="val").sequences] == [val_clip]
    assert [s.name for s in load_chimpact(root=tmp_path, split="test").sequences] == [test_clip]
    assert [s.name for s in load_chimpact(root=tmp_path, split="train").sequences] == [train_clip]


def _write_seqinfo(seq_dir: Path, seq_length: int | str) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq_dir.name}\nimDir=img1\nframeRate=20\n"
        f"seqLength={seq_length}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )


def _write_mc_gt(seq_dir: Path, rows: list[str]) -> None:
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "gt.txt").write_text("\n".join(rows) + "\n" if rows else "")


def _make_mc_sequence(
    root: Path, split: str, seq_name: str, seq_length: int | str, rows: list[str]
) -> Path:
    seq_dir = root / split / seq_name
    _write_seqinfo(seq_dir, seq_length)
    _write_mc_gt(seq_dir, rows)
    return seq_dir


def test_motchallenge_sequence_discovery_sorted_by_name(tmp_path):
    for name in ("seq-02", "seq-10", "seq-01"):
        _make_mc_sequence(tmp_path, "val", name, 1, ["1,1,10,10,20,20,1,1,1"])

    dataset = load_motchallenge(tmp_path, "val")

    assert [seq.name for seq in dataset.sequences] == ["seq-01", "seq-02", "seq-10"]


def test_motchallenge_seq_length_comes_from_seqinfo_not_max_gt_frame(tmp_path):
    _make_mc_sequence(
        tmp_path,
        "val",
        "seq-01",
        seq_length=10,
        rows=[
            "1,1,10,10,20,20,1,1,1",
            "2,1,12,10,20,20,1,1,1",
            "3,1,14,10,20,20,1,1,1",
        ],
    )

    dataset = load_motchallenge(tmp_path, "val")

    (seq,) = dataset.sequences
    assert seq.num_timesteps == 10


def test_motchallenge_missing_seqinfo_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    _write_mc_gt(seq_dir, ["1,1,10,10,20,20,1,1,1"])

    with pytest.raises(ValueError, match="seqinfo.ini"):
        load_motchallenge(tmp_path, "val")


def test_motchallenge_seqinfo_missing_seqlength_key_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    seq_dir.mkdir(parents=True)
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=seq-01\nframeRate=20\n")
    _write_mc_gt(seq_dir, ["1,1,10,10,20,20,1,1,1"])

    with pytest.raises(ValueError, match="seqLength"):
        load_motchallenge(tmp_path, "val")


def test_motchallenge_seqinfo_non_integer_seqlength_raises(tmp_path):
    _make_mc_sequence(
        tmp_path, "val", "seq-01", seq_length="not-a-number", rows=["1,1,10,10,20,20,1,1,1"]
    )

    with pytest.raises(ValueError, match="seqLength"):
        load_motchallenge(tmp_path, "val")


def test_motchallenge_missing_gt_txt_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    _write_seqinfo(seq_dir, 5)

    with pytest.raises(ValueError, match="gt.txt"):
        load_motchallenge(tmp_path, "val")


def test_motchallenge_malformed_gt_row_raises_and_names_file(tmp_path):
    seq_dir = _make_mc_sequence(
        tmp_path, "val", "seq-01", seq_length=2, rows=["1,1,10,10,20,20,1,1,1", "notaframe,1,1"]
    )

    with pytest.raises(ValueError) as exc:
        load_motchallenge(tmp_path, "val")
    assert str(seq_dir / "gt" / "gt.txt") in str(exc.value)


def test_motchallenge_frames_binned_regardless_of_file_order(tmp_path):
    _make_mc_sequence(
        tmp_path,
        "val",
        "seq-01",
        seq_length=3,
        rows=[
            "3,1,14,10,20,20,1,1,1",
            "1,1,10,10,20,20,1,1,1",
            "2,1,12,10,20,20,1,1,1",
        ],
    )

    dataset = load_motchallenge(tmp_path, "val")
    (seq,) = dataset.sequences
    pred_dir = tmp_path / "pred"
    write_mot(pred_dir / "seq-01.txt", [t for t in seq.tracks])

    clear = evaluate(dataset, pred_dir, [CLEAR()]).combined["CLEAR"]
    assert seq.num_timesteps == 3
    assert (clear["CLR_TP"], clear["MOTP"]) == (3.0, 1.0)


def _uavdt_gt_dir(root: Path) -> Path:
    return root / "UAV-benchmark-MOTD_v1.0" / "GT"


def _write_uavdt_gt(root: Path, seq: str, rows: list[str]) -> None:
    gt_dir = _uavdt_gt_dir(root)
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{seq}_gt.txt").write_text("\n".join(rows) + "\n")


def _write_uavdt_ignore(root: Path, seq: str, rows: list[str]) -> None:
    gt_dir = _uavdt_gt_dir(root)
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{seq}_gt_ignore.txt").write_text("\n".join(rows) + "\n")


def test_uavdt_sequence_discovery_excludes_ignore_and_whole_files(tmp_path):
    _write_uavdt_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    _write_uavdt_ignore(tmp_path, "M0101", ["1,9,100,100,50,50,1,-1,-1"])
    (_uavdt_gt_dir(tmp_path) / "M0101_gt_whole.txt").write_text("1,1,0,0,10,10,1,1,3\n")

    dataset = load_uavdt(root=tmp_path, split="all")

    assert [s.name for s in dataset.sequences] == ["M0101"]


def test_uavdt_unknown_split_raises(tmp_path):
    _write_uavdt_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])

    with pytest.raises(ValueError, match="unknown uavdt split"):
        load_uavdt(root=tmp_path, split="test")


def test_uavdt_malformed_ignore_row_raises(tmp_path):
    _write_uavdt_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    _write_uavdt_ignore(tmp_path, "M0101", ["notaframe,9,100,100,50,50,1,-1,-1"])

    with pytest.raises(ValueError, match="malformed MOT row"):
        load_uavdt(root=tmp_path, split="all")


def test_uavdt_ignore_file_parses_into_gt_sequence_ignore_regions(tmp_path):
    _write_uavdt_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    _write_uavdt_ignore(tmp_path, "M0101", ["1,9,100,100,50,50,1,-1,-1"])

    dataset = load_uavdt(root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert len(seq.ignore_regions) == 1
    region = seq.ignore_regions[0]
    assert (region.frame, region.x, region.y, region.w, region.h) == (1, 100.0, 100.0, 50.0, 50.0)


def test_uavdt_missing_ignore_file_yields_no_ignore_regions(tmp_path):
    _write_uavdt_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])

    dataset = load_uavdt(root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert seq.ignore_regions == ()


def test_uavdt_prediction_inside_ignore_region_excluded_through_evaluate(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    _write_uavdt_gt(gt_root, "M0101", ["1,1,0,0,10,10,1,1,-1", "2,1,1,0,10,10,1,1,-1"])
    _write_uavdt_ignore(
        gt_root, "M0101", ["1,9,100,100,50,50,1,-1,-1", "2,9,100,100,50,50,1,-1,-1"]
    )

    dataset = load_uavdt(root=gt_root, split="all")
    (seq,) = dataset.sequences
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [
            Track(frame=1, track_id=901, x=110, y=110, w=20, h=20, conf=0.9),
            Track(frame=1, track_id=902, x=300, y=300, w=20, h=20, conf=0.9),
        ],
    )

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }


def test_dancetrack_loads_with_explicit_root(tmp_path):
    _make_mc_sequence(
        tmp_path, "val", "dancetrack0001", seq_length=2, rows=["1,1,10,10,20,20,1,1,1"]
    )

    dataset = load_dancetrack(root=tmp_path, split="val")

    assert dataset.name == "dancetrack"
    assert [seq.name for seq in dataset.sequences] == ["dancetrack0001"]


def test_sportsmot_loads_with_explicit_root(tmp_path):
    _make_mc_sequence(tmp_path, "val", "v_00001", seq_length=2, rows=["1,1,10,10,20,20,1,1,1"])

    dataset = load_sportsmot(root=tmp_path, split="val")

    assert dataset.name == "sportsmot"
    assert [seq.name for seq in dataset.sequences] == ["v_00001"]


def test_load_dataset_reads_benchmark_under_moteval_data_root(tmp_path, monkeypatch):
    _make_mc_sequence(
        tmp_path / "dancetrack", "val", "env-root-seq", seq_length=3, rows=["1,1,10,10,20,20,1"]
    )
    monkeypatch.setenv("MOTEVAL_DATA_ROOT", str(tmp_path))

    dataset = load_dataset("dancetrack")

    assert dataset.split == "val"
    assert [(seq.name, seq.num_timesteps) for seq in dataset.sequences] == [("env-root-seq", 3)]


@pytest.mark.parametrize("value", ["", "   "])
def test_load_dataset_rejects_blank_moteval_data_root(value, monkeypatch):
    monkeypatch.setenv("MOTEVAL_DATA_ROOT", value)

    with pytest.raises(ValueError, match="MOTEVAL_DATA_ROOT must not be empty"):
        load_dataset("dancetrack")


def _write_panaf500_ann(root: Path, split: str, video_id: str, annotations: list[dict]) -> None:
    ann_dir = root / "annotations" / split
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{video_id}.json").write_text(
        json.dumps({"video": video_id, "annotations": annotations})
    )


def test_panaf500_gt_boxes_converted_from_xyxy_to_xywh(tmp_path):
    _write_panaf500_ann(
        tmp_path,
        "validation",
        "vid1",
        [
            {"frame_id": 1, "detections": [{"bbox": [10, 20, 60, 120], "ape_id": 0}]},
            {"frame_id": 2, "detections": []},
            {
                "frame_id": 3,
                "detections": [
                    {"bbox": [0, 0, 10, 10], "ape_id": 1},
                    {"bbox": [5.5, 6.5, 16.0, 27.25], "ape_id": 2},
                ],
            },
        ],
    )

    dataset = load_panaf500(root=tmp_path, split="validation")

    (seq,) = dataset.sequences
    assert seq.name == "vid1"
    assert len(seq.tracks) == 3
    boxes = {t.track_id: (t.frame, t.x, t.y, t.w, t.h) for t in seq.tracks}
    assert boxes == {
        0: (1, 10, 20, 50, 100),
        1: (3, 0, 0, 10, 10),
        2: (3, 5.5, 6.5, 10.5, 20.75),
    }
    assert seq.num_timesteps == 3


def test_panaf500_missing_split_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="split directory not found"):
        load_panaf500(root=tmp_path, split="validation")


def test_panaf500_empty_annotations_raises_on_seq_length_derivation(tmp_path):
    _write_panaf500_ann(tmp_path, "validation", "vid1", [])

    with pytest.raises(ValueError, match="empty gt"):
        load_panaf500(root=tmp_path, split="validation")


def test_panaf500_end_to_end_evaluate_with_independently_numbered_predictions(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()

    _write_panaf500_ann(
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
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=1, track_id=900, x=11, y=21, w=50, h=100, conf=0.9)],
    )

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }


def test_load_motchallenge_scores_a_custom_layout_without_registration(tmp_path):
    root = tmp_path / "my-tracker-data"
    seq_dir = root / "train" / "SEQ01"
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "gt" / "gt.txt").write_text("1,1,10,10,20,20,1\n2,1,12,10,20,20,1\n")
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nseqLength=2\n")

    dataset = load_motchallenge(root)

    assert dataset.name == "my-tracker-data"
    assert dataset.split == "train"

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    write_mot(
        pred_dir / "SEQ01.txt",
        [Track(frame=1, track_id=901, x=10, y=10, w=20, h=20, conf=0.9)],
    )
    result = evaluate(dataset, pred_dir, [Count()])
    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }


def test_load_motchallenge_drops_conf_zero_gt_rows(tmp_path):
    root = tmp_path / "custom"
    seq_dir = root / "train" / "SEQ01"
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "gt" / "gt.txt").write_text("1,1,10,10,20,20,1\n1,2,100,100,20,20,0\n")
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nseqLength=1\n")

    dataset = load_motchallenge(root)
    pred_dir = tmp_path / "no-preds"
    pred_dir.mkdir()
    (pred_dir / "SEQ01.txt").write_text("")
    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"]["GT_Dets"] == 1.0
