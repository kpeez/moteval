from dataclasses import replace
from pathlib import Path

import pytest

from moteval import evaluate
from moteval.benchmarks.gmot40 import GMOT40_PROTOCOL, load_gmot40
from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention, GtSequence
from moteval.formats.mot_txt import Track, write_mot
from moteval.metrics.count import Count

_CATEGORIES = ("airplane", "ball", "bird", "car", "fish", "insect", "person", "stock")
_ANIMAL_CATEGORIES = {"bird", "fish", "insect", "stock"}


def _write_gt(root: Path, seq: str, rows: list[str]) -> None:
    gt_dir = root / "track_label"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{seq}.txt").write_text("\n".join(rows) + "\n")


def test_sequence_discovery_and_animal_subset(tmp_path):
    for category in _CATEGORIES:
        _write_gt(tmp_path, f"{category}-0", ["0,1,10,10,20,20,1,1,1"])

    dataset = load_gmot40(root=tmp_path, split="test")
    assert len(dataset.sequences) == len(_CATEGORIES)

    animal = load_gmot40(root=tmp_path, split="animal")
    assert {s.name for s in animal.sequences} == {f"{c}-0" for c in _ANIMAL_CATEGORIES}


def test_unknown_split_raises(tmp_path):
    _write_gt(tmp_path, "bird-0", ["0,1,10,10,20,20,1,1,1"])
    with pytest.raises(ValueError, match="unknown gmot40 split"):
        load_gmot40(root=tmp_path, split="nope")


def test_seq_length_derived_from_last_annotated_zero_indexed_frame(tmp_path):
    _write_gt(
        tmp_path,
        "bird-0",
        ["0,1,10,20,30,40,1,1,1", "2,1,12,22,30,40,1,1,1"],
    )

    dataset = load_gmot40(root=tmp_path, split="test")

    (seq,) = dataset.sequences
    assert seq.num_timesteps == 3


def test_frame_zero_contributes_and_matches_one_indexed_reencoding(tmp_path):
    _write_gt(
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


def test_end_to_end_evaluate_with_independently_numbered_predictions(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()

    _write_gt(
        gt_root,
        "bird-0",
        ["0,1,10,10,20,20,1,1,1", "1,1,12,10,20,20,1,1,1"],
    )

    dataset = load_gmot40(root=gt_root, split="test")
    (seq,) = dataset.sequences
    # Predictions numbered independently of GT: different track id, and only a
    # disjoint one-frame subset of the two GT frames (frame 0, not frame 1).
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=0, track_id=901, x=11, y=11, w=20, h=20, conf=0.9)],
    )

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }
