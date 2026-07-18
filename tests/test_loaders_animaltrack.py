from pathlib import Path

import pytest

from moteval import evaluate
from moteval.benchmarks.animaltrack import ANIMALTRACK_PROTOCOL, load_animaltrack
from moteval.formats.mot_txt import write_mot
from moteval.metrics.count import Count


def _write_gt(root: Path, seq: str, rows: list[str]) -> None:
    gt_dir = root / "gt_all"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{seq}_gt.txt").write_text("\n".join(rows) + "\n")


def _write_split_file(root: Path, split: str, seqs: list[str]) -> None:
    split_dir = root / "train_test_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(f"{seq}.mp4" for seq in seqs) + "\n"
    (split_dir / f"videos_{split}.txt").write_text(content)


def test_all_split_discovers_every_gt_file(tmp_path):
    for seq in ("chicken_1", "deer_2"):
        _write_gt(tmp_path, seq, ["1,1,10,10,20,20,1,1,1"])

    dataset = load_animaltrack(root=tmp_path, split="all")

    assert [s.name for s in dataset.sequences] == ["chicken_1", "deer_2"]


def test_train_test_selection_from_split_file(tmp_path):
    _write_gt(tmp_path, "chicken_1", ["1,1,10,10,20,20,1,1,1"])
    _write_gt(tmp_path, "deer_2", ["1,1,10,10,20,20,1,1,1"])
    _write_split_file(tmp_path, "train", ["chicken_1"])
    _write_split_file(tmp_path, "test", ["deer_2"])

    train = load_animaltrack(root=tmp_path, split="train")
    test = load_animaltrack(root=tmp_path, split="test")

    assert [s.name for s in train.sequences] == ["chicken_1"]
    assert [s.name for s in test.sequences] == ["deer_2"]


def test_seq_length_derived_from_last_annotated_frame(tmp_path):
    _write_gt(
        tmp_path,
        "chicken_1",
        ["1,1,22,281,126,196,1,1,1", "3,1,25,283,126,196,1,1,1"],
    )

    dataset = load_animaltrack(root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert seq.num_timesteps == 3


def test_missing_split_file_raises(tmp_path):
    (tmp_path / "gt_all").mkdir()
    with pytest.raises(ValueError, match="not found at"):
        load_animaltrack(root=tmp_path, split="train")


def test_unknown_split_raises(tmp_path):
    (tmp_path / "gt_all").mkdir()
    with pytest.raises(ValueError, match="unknown animaltrack split"):
        load_animaltrack(root=tmp_path, split="nope")


def test_gt_class_id_matches_protocol_eval_class(tmp_path):
    # Regression for the class_id hazard (issue #11 comment): every GT Track
    # must be stamped with the protocol's evaluated class, never left to
    # read_mot's silent default.
    _write_gt(tmp_path, "chicken_1", ["1,1,10,10,20,20,1,1,1"])

    dataset = load_animaltrack(root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert all(track.class_id in ANIMALTRACK_PROTOCOL.eval_classes for track in seq.tracks)


def test_end_to_end_evaluate_with_independently_numbered_predictions(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()

    _write_gt(
        gt_root,
        "chicken_1",
        ["1,1,10,10,20,20,1,1,1", "2,1,12,10,20,20,1,1,1"],
    )

    dataset = load_animaltrack(root=gt_root, split="all")
    (seq,) = dataset.sequences
    write_mot(pred_dir / f"{seq.name}.txt", list(seq.tracks))

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 2.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }
