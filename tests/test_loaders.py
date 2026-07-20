"""Per-benchmark loader tests: common MOTChallenge-adapter machinery is
parameterized over a representative matrix; everything below is a targeted
quirk test for one loader's documented deviation from the default layout.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from moteval import evaluate
from moteval.benchmarks.animaltrack import load_animaltrack
from moteval.benchmarks.bft import load_bft
from moteval.benchmarks.chimpact import _CLASS_ID, _TEST_CLIPS, _VAL_CLIPS, load_chimpact
from moteval.benchmarks.dancetrack import load_dancetrack
from moteval.benchmarks.gmot40 import GMOT40_PROTOCOL, load_gmot40
from moteval.benchmarks.motchallenge import MOTChallengeConfig, load_motchallenge
from moteval.benchmarks.panaf500 import _CLASS_ID as _PANAF500_CLASS_ID
from moteval.benchmarks.panaf500 import _xyxy_to_xywh, load_panaf500
from moteval.benchmarks.sportsmot import load_sportsmot
from moteval.benchmarks.uavdt import UAVDT_CONFIG, UAVDT_PROTOCOL, load_uavdt
from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention, GtSequence
from moteval.data.protocol import Protocol
from moteval.formats import Track, write_mot
from moteval.metrics.count import Count

# ------------------------------------------------------------ common matrix
#
# BFT, AnimalTrack, and GMOT-40 all reuse the MOTChallengeConfig adapter with
# a flat, seqinfo-free GT directory. Each entry provides a GT writer matching
# its loader's real layout and the loader itself, so sequence discovery, last-
# annotated-frame seq_length derivation, and end-to-end evaluate are proven
# once per dataset without repeating the same assertions three times over.


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
    # (loader, write_gt, seq_name, last_frame_offset) -- offset is added to the
    # last annotated frame number to get num_timesteps: 0 for 1-indexed
    # loaders (bft, animaltrack), 1 for gmot40's native 0-indexed frames.
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
    # Predictions numbered independently of GT: different track id, disjoint
    # single-frame box (not a copy of any GT row).
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=seq.tracks[0].frame, track_id=901, x=11, y=11, w=20, h=20, conf=0.9)],
    )

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }


# ---------------------------------------------------------------- animaltrack


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


# ----------------------------------------------------------------------- bft


def test_bft_missing_split_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="split directory not found"):
        load_bft(root=tmp_path, split="val")


def test_bft_empty_gt_raises_on_seq_length_derivation(tmp_path):
    _write_bft_gt(tmp_path, "seqA", [])
    with pytest.raises(ValueError, match="empty gt"):
        load_bft(root=tmp_path, split="val")


# ---------------------------------------------------------- BFT/AnimalTrack/GMOT-40
# num_timesteps undercounts when a sequence has no GT in its final frames --
# harmless for metrics, but a prediction past the last annotated frame raises.


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


# --------------------------------------------------------------------- gmot40


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


# ---------------------------------------------------- GMOT-40 + ChimpACT: 0-indexing
#
# Both are natively 0-indexed and both loaders keep raw 0-indexed frame
# numbers, declaring FrameConvention(first_frame=0), rather than shifting like
# the legacy loader did (ADR-0002).


def test_gmot40_frame_zero_contributes_and_matches_one_indexed_reencoding(tmp_path):
    _write_gmot40_gt(
        tmp_path,
        "bird-0",
        ["0,1,10,20,30,40,1,1,1", "1,1,12,22,30,40,1,1,1"],
    )

    dataset = load_gmot40(root=tmp_path, split="test")
    (seq,) = dataset.sequences
    native_pred = tuple(seq.tracks)

    native_data = build_sequence_data(seq, native_pred, GMOT40_PROTOCOL, 1)
    # Frame 0 (the first timestep) must contribute -- the silent-drop bug this
    # rewrite exists to kill would have discarded it.
    assert native_data.gt_ids[0].shape[0] == 1
    assert native_data.num_gt_dets == 2

    one_indexed_convention = FrameConvention(name="1-indexed", first_frame=1)
    one_indexed_protocol = replace(GMOT40_PROTOCOL, frame_convention=one_indexed_convention)
    shifted_gt = GtSequence(
        name=seq.name,
        num_timesteps=seq.num_timesteps,
        tracks=tuple(replace(t, frame=t.frame + 1) for t in seq.tracks),
    )
    shifted_pred = tuple(replace(t, frame=t.frame + 1) for t in native_pred)
    shifted_data = build_sequence_data(shifted_gt, shifted_pred, one_indexed_protocol, 1)

    count = Count()
    assert count.eval_sequence(native_data) == count.eval_sequence(shifted_data)
    for native_frame, shifted_frame in zip(
        native_data.gt_boxes, shifted_data.gt_boxes, strict=True
    ):
        assert native_frame.tolist() == shifted_frame.tolist()


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

    assert 0 in {t.frame for t in seq.tracks}
    frame_zero = next(t for t in seq.tracks if t.frame == 0)
    assert (frame_zero.x, frame_zero.y, frame_zero.w, frame_zero.h) == (1.0, 2.0, 3.0, 4.0)

    data = build_sequence_data(seq, seq.tracks, dataset.protocol, _CLASS_ID)
    # Frame 0 is the first timestep -- the silent-drop bug this rewrite
    # exists to kill would have discarded it.
    assert data.gt_ids[0].shape[0] == 1


# ------------------------------------------------------------------ chimpact
#
# Interpolation/hold semantics replicate legacy track-zoo exactly (issue #12
# decision): comparability with arXiv:2511.02591's published baselines
# requires matching legacy output even where a "cleaner" rule would diverge.


def test_chimpact_interpolation_matches_hand_derived_fixture(tmp_path):
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
    # Fractional intermediate frames are the interesting hand-checked cases.
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
    # Ported from legacy track-zoo's test_missing_next_keyframe_holds_last_box.
    # A single keyframe block for the whole clip holds its box constant
    # through all 9 interior frames that follow it -- there is no next block
    # to interpolate toward.
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
    _write_chimpact_clip(tmp_path, "clip-d", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert seq.num_timesteps == 30

    track5_frames = sorted(t.frame for t in seq.tracks if t.track_id == 5)
    assert track5_frames == list(range(0, 20))
    held = [t for t in seq.tracks if t.track_id == 5 and 11 <= t.frame <= 19]
    assert all((t.x, t.y, t.w, t.h) == (10.0, 10.0, 10.0, 10.0) for t in held)

    track7_frames = sorted(t.frame for t in seq.tracks if t.track_id == 7)
    assert track7_frames == list(range(10, 30))
    tail = [t for t in seq.tracks if t.track_id == 7 and 21 <= t.frame <= 29]
    assert all((t.x, t.y, t.w, t.h) == (10.0, 10.0, 5.0, 5.0) for t in tail)


def test_chimpact_missing_intervening_keyframe_block_yields_zero_rows_for_that_window(tmp_path):
    # Ported from legacy's test_frame_with_no_keyframe_block_has_no_gt_rows,
    # with the missing block sandwiched between two real keyframes: block 1
    # has no image entry at all, so frames 10-19 get zero GT rows for any
    # track, regardless of which tracks are alive on either side of the gap.
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
    assert sorted(t.frame for t in track5) == list(range(0, 10))
    assert all((t.x, t.y, t.w, t.h) == (1.0, 1.0, 2.0, 2.0) for t in track5)

    track9 = [t for t in seq.tracks if t.track_id == 9]
    assert sorted(t.frame for t in track9) == list(range(20, 30))
    assert all((t.x, t.y, t.w, t.h) == (3.0, 3.0, 4.0, 4.0) for t in track9)


def test_chimpact_multi_block_gap_then_reappearance_has_no_back_connection(tmp_path):
    # Track 5 lives at block 0, is absent from blocks 1 and 2 (a different
    # track, 9, exists there), then reappears at block 3 with an unrelated
    # box. Legacy holds for exactly one block after death, is silent through
    # the entire multi-block gap, then treats the reappearance as a fresh,
    # unconnected keyframe -- no interpolation back to its old position.
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
    assert track5_frames == list(range(0, 10)) + list(range(30, 40))
    before_gap = [t for t in seq.tracks if t.track_id == 5 and t.frame < 10]
    assert all((t.x, t.y, t.w, t.h) == (1.0, 1.0, 2.0, 2.0) for t in before_gap)
    after_gap = [t for t in seq.tracks if t.track_id == 5 and t.frame >= 30]
    assert all((t.x, t.y, t.w, t.h) == (100.0, 100.0, 6.0, 6.0) for t in after_gap)


def test_chimpact_gt_class_id_matches_declared_constant_not_raw_coco_category(tmp_path):
    # Regression for the class_id hazard (issue #12 review comment): this
    # loader never touches read_mot, so every Track must be stamped
    # explicitly with the module's declared class. Each fixture annotation
    # also carries the real ChimpACT COCO schema's `category_id` (always 0,
    # "Chimpanzee"), so a loader that blindly forwarded that raw field
    # instead would produce `class_id == 0` here, not 1.
    labels = {
        "images": [_coco_image(0, 0)],
        "annotations": [_coco_ann(0, 0, 1, [0.0, 0.0, 10.0, 10.0])],
    }
    _write_chimpact_clip(tmp_path, "clip-e", labels)

    dataset = load_chimpact(root=tmp_path, split="train")
    (seq,) = dataset.sequences

    assert all(track.class_id == _CLASS_ID for track in seq.tracks)
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
    val_clip = _VAL_CLIPS[0]
    test_clip = _TEST_CLIPS[0]
    train_clip = "not-an-official-clip-name"
    tiny_labels = {"images": [_coco_image(0, 0)], "annotations": [_coco_ann(0, 0, 1, [0, 0, 1, 1])]}
    for clip in (val_clip, test_clip, train_clip):
        _write_chimpact_clip(tmp_path, clip, tiny_labels)

    assert [s.name for s in load_chimpact(root=tmp_path, split="val").sequences] == [val_clip]
    assert [s.name for s in load_chimpact(root=tmp_path, split="test").sequences] == [test_clip]
    assert [s.name for s in load_chimpact(root=tmp_path, split="train").sequences] == [train_clip]


# ---------------------------------------------------------------- motchallenge


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


_MC_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
_MC_TEST_PROTOCOL = Protocol(
    name="test-motchallenge", frame_convention=_MC_CONVENTION, eval_classes=(1,)
)


def _mc_config(root: Path) -> MOTChallengeConfig:
    return MOTChallengeConfig(
        name="test-motchallenge", default_root=root, protocol=_MC_TEST_PROTOCOL
    )


def test_motchallenge_sequence_discovery_sorted_by_name(tmp_path):
    for name in ("seq-02", "seq-10", "seq-01"):
        _make_mc_sequence(tmp_path, "val", name, 1, ["1,1,10,10,20,20,1,1,1"])

    dataset = load_motchallenge(_mc_config(tmp_path), split="val")

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

    dataset = load_motchallenge(_mc_config(tmp_path), split="val")

    (seq,) = dataset.sequences
    assert seq.num_timesteps == 10


def test_motchallenge_missing_seqinfo_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    _write_mc_gt(seq_dir, ["1,1,10,10,20,20,1,1,1"])

    with pytest.raises(ValueError, match="seqinfo.ini"):
        load_motchallenge(_mc_config(tmp_path), split="val")


def test_motchallenge_seqinfo_missing_seqlength_key_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    seq_dir.mkdir(parents=True)
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=seq-01\nframeRate=20\n")
    _write_mc_gt(seq_dir, ["1,1,10,10,20,20,1,1,1"])

    with pytest.raises(ValueError, match="seqLength"):
        load_motchallenge(_mc_config(tmp_path), split="val")


def test_motchallenge_seqinfo_non_integer_seqlength_raises(tmp_path):
    _make_mc_sequence(
        tmp_path, "val", "seq-01", seq_length="not-a-number", rows=["1,1,10,10,20,20,1,1,1"]
    )

    with pytest.raises(ValueError, match="seqLength"):
        load_motchallenge(_mc_config(tmp_path), split="val")


def test_motchallenge_missing_gt_txt_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    _write_seqinfo(seq_dir, 5)

    with pytest.raises(ValueError, match="gt.txt"):
        load_motchallenge(_mc_config(tmp_path), split="val")


def test_motchallenge_malformed_gt_row_raises_and_names_file(tmp_path):
    seq_dir = _make_mc_sequence(
        tmp_path, "val", "seq-01", seq_length=2, rows=["1,1,10,10,20,20,1,1,1", "notaframe,1,1"]
    )

    with pytest.raises(ValueError) as exc:
        load_motchallenge(_mc_config(tmp_path), split="val")
    assert str(seq_dir / "gt" / "gt.txt") in str(exc.value)


def test_motchallenge_gt_class_id_explicitly_stamped_from_config(tmp_path):
    # read_mot defaults every row to class_id=1, so this only passes if
    # `_load_sequence` actually stamps `config.class_id` rather than passing
    # rows through untouched.
    _make_mc_sequence(
        tmp_path,
        "val",
        "seq-01",
        seq_length=2,
        rows=["1,1,10,10,20,20,1,1,1", "2,1,12,10,20,20,1,1,1"],
    )
    config = MOTChallengeConfig(
        name="test-motchallenge-class7",
        default_root=tmp_path,
        protocol=Protocol(name="test-class7", frame_convention=_MC_CONVENTION, eval_classes=(7,)),
        class_id=7,
    )

    dataset = load_motchallenge(config, split="val")

    (seq,) = dataset.sequences
    assert all(track.class_id == 7 for track in seq.tracks)


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

    dataset = load_motchallenge(_mc_config(tmp_path), split="val")
    (seq,) = dataset.sequences
    data = build_sequence_data(seq, (), _MC_TEST_PROTOCOL, 1)

    assert data.num_timesteps == 3
    assert [frame.shape[0] for frame in data.gt_ids] == [1, 1, 1]


# --------------------------------------------------------------------- uavdt


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


def test_uavdt_gt_class_id_explicitly_stamped_from_config(tmp_path):
    _write_uavdt_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    config = replace(
        UAVDT_CONFIG,
        default_root=tmp_path,
        class_id=7,
        protocol=replace(UAVDT_PROTOCOL, eval_classes=(7,)),
    )

    dataset = load_motchallenge(config, root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert all(track.class_id == 7 for track in seq.tracks)


def test_uavdt_ignore_file_parses_into_gt_sequence_ignore_regions(tmp_path):
    _write_uavdt_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    _write_uavdt_ignore(tmp_path, "M0101", ["1,9,100,100,50,50,1,-1,-1"])

    dataset = load_uavdt(root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert len(seq.ignore_regions) == 1
    region = seq.ignore_regions[0]
    assert (region.frame, region.x, region.y, region.w, region.h) == (1, 100.0, 100.0, 50.0, 50.0)


def test_uavdt_missing_ignore_file_yields_no_ignore_regions(tmp_path):
    # Legacy layout has no marker distinguishing "deliberately no ignore
    # regions" from "file just doesn't exist" -- absence means no regions.
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
            # Fully inside the ignore region, unmatched to any GT: excluded.
            Track(frame=1, track_id=901, x=110, y=110, w=20, h=20, conf=0.9),
            # Far from GT and the ignore region, unmatched: kept.
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


def test_uavdt_ignore_regions_change_evaluation_results(tmp_path):
    # Control: identical GT and predictions, with vs. without the ignore file,
    # must produce different Count Dets -- proves the regions have effect.
    def _build_gt(root: Path, with_ignore: bool) -> None:
        _write_uavdt_gt(root, "M0101", ["1,1,0,0,10,10,1,1,-1", "2,1,1,0,10,10,1,1,-1"])
        if with_ignore:
            _write_uavdt_ignore(
                root, "M0101", ["1,9,100,100,50,50,1,-1,-1", "2,9,100,100,50,50,1,-1,-1"]
            )

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    write_mot(
        pred_dir / "M0101.txt",
        [
            Track(frame=1, track_id=901, x=110, y=110, w=20, h=20, conf=0.9),
            Track(frame=1, track_id=902, x=300, y=300, w=20, h=20, conf=0.9),
        ],
    )

    honored_root = tmp_path / "honored"
    _build_gt(honored_root, with_ignore=True)
    honored = evaluate(load_uavdt(root=honored_root, split="all"), pred_dir, [Count()])

    control_root = tmp_path / "control"
    _build_gt(control_root, with_ignore=False)
    control = evaluate(load_uavdt(root=control_root, split="all"), pred_dir, [Count()])

    assert control.combined["Count"]["Dets"] == 2.0
    assert honored.combined["Count"]["Dets"] == 1.0


# ------------------------------------------------------ dancetrack / sportsmot
#
# Both reuse the default MOTChallengeConfig layout unmodified; only the root
# override and registered dataset name need proving per benchmark.


def test_dancetrack_root_override_does_not_touch_default_root(tmp_path):
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


# -------------------------------------------------------------------- panaf500


def _write_panaf500_ann(root: Path, split: str, video_id: str, annotations: list[dict]) -> None:
    ann_dir = root / "annotations" / split
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{video_id}.json").write_text(
        json.dumps({"video": video_id, "annotations": annotations})
    )


def test_panaf500_xyxy_to_xywh_hand_computed():
    assert _xyxy_to_xywh([10, 20, 60, 120]) == (10, 20, 50, 100)
    assert _xyxy_to_xywh([0, 0, 1, 1]) == (0, 0, 1, 1)
    assert _xyxy_to_xywh([5.5, 6.5, 15.5, 26.5]) == (5.5, 6.5, 10.0, 20.0)


def test_panaf500_gt_boxes_converted_from_xyxy_to_xywh(tmp_path):
    _write_panaf500_ann(
        tmp_path,
        "validation",
        "vid1",
        [
            {"frame_id": 1, "detections": [{"bbox": [10, 20, 60, 120], "ape_id": 0}]},
            {"frame_id": 2, "detections": []},
            {"frame_id": 3, "detections": [{"bbox": [0, 0, 10, 10], "ape_id": 1}]},
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


def test_panaf500_missing_split_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="split directory not found"):
        load_panaf500(root=tmp_path, split="validation")


def test_panaf500_empty_annotations_raises_on_seq_length_derivation(tmp_path):
    # Consistent with BFT/AnimalTrack/GMOT-40: no frame-count source other than
    # the annotations themselves, so an empty video raises loudly rather than
    # silently defaulting to num_timesteps=0.
    _write_panaf500_ann(tmp_path, "validation", "vid1", [])

    with pytest.raises(ValueError, match="empty gt"):
        load_panaf500(root=tmp_path, split="validation")


def test_panaf500_gt_class_id_matches_module_declared_constant(tmp_path):
    # Regression for the class_id hazard: this loader never touches read_mot,
    # so every Track must be stamped explicitly with the module's declared
    # class rather than left to the dataclass default.
    _write_panaf500_ann(
        tmp_path,
        "validation",
        "vid1",
        [{"frame_id": 1, "detections": [{"bbox": [0, 0, 10, 10], "ape_id": 0}]}],
    )

    dataset = load_panaf500(root=tmp_path, split="validation")

    (seq,) = dataset.sequences
    assert all(track.class_id == _PANAF500_CLASS_ID for track in seq.tracks)


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
    # Predictions numbered independently of GT: different track id, and only a
    # disjoint one-frame subset of the two GT frames.
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
