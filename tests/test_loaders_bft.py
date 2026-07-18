from pathlib import Path

import pytest

from moteval import evaluate
from moteval.benchmarks.bft import load_bft
from moteval.formats.mot_txt import Track, write_mot
from moteval.metrics.count import Count


def _write_gt(root: Path, split: str, seq: str, rows: list[str]) -> None:
    ann_dir = root / "annotations_mot" / split
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{seq}.txt").write_text("\n".join(rows) + "\n")


def test_sequence_discovery_from_flat_annotation_dir(tmp_path):
    for seq in ("seqA", "seqB"):
        _write_gt(tmp_path, "val", seq, ["1,1,10,10,20,20,1,1,1"])

    dataset = load_bft(root=tmp_path, split="val")

    assert [s.name for s in dataset.sequences] == ["seqA", "seqB"]


def test_seq_length_derived_from_last_annotated_frame(tmp_path):
    _write_gt(
        tmp_path,
        "val",
        "seqA",
        [
            "1,1,10,10,20,20,1,1,1",
            "2,1,12,10,20,20,1,1,1",
            "3,2,5,5,10,10,1,1,1",
        ],
    )

    dataset = load_bft(root=tmp_path, split="val")

    (seq,) = dataset.sequences
    assert seq.num_timesteps == 3


def test_missing_split_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="split directory not found"):
        load_bft(root=tmp_path, split="val")


def test_empty_gt_raises_on_seq_length_derivation(tmp_path):
    _write_gt(tmp_path, "val", "seqA", [])

    with pytest.raises(ValueError, match="empty gt"):
        load_bft(root=tmp_path, split="val")


def test_end_to_end_evaluate_with_independently_numbered_predictions(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()

    _write_gt(
        gt_root,
        "val",
        "seqA",
        [
            "1,1,10,10,20,20,1,1,1",
            "2,1,12,10,20,20,1,1,1",
        ],
    )

    dataset = load_bft(root=gt_root, split="val")
    (seq,) = dataset.sequences
    # Predictions numbered independently of GT: different track id, and only a
    # disjoint one-frame subset of the two GT frames.
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=2, track_id=901, x=13, y=11, w=20, h=20, conf=0.9)],
    )

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }
