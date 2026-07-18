from dataclasses import replace
from pathlib import Path

import pytest

from moteval import evaluate
from moteval.benchmarks.motchallenge import load_motchallenge
from moteval.benchmarks.uavdt import UAVDT_CONFIG, UAVDT_PROTOCOL, load_uavdt
from moteval.formats.mot_txt import Track, write_mot
from moteval.metrics.count import Count


def _gt_dir(root: Path) -> Path:
    return root / "UAV-benchmark-MOTD_v1.0" / "GT"


def _write_gt(root: Path, seq: str, rows: list[str]) -> None:
    gt_dir = _gt_dir(root)
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{seq}_gt.txt").write_text("\n".join(rows) + "\n")


def _write_ignore(root: Path, seq: str, rows: list[str]) -> None:
    gt_dir = _gt_dir(root)
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{seq}_gt_ignore.txt").write_text("\n".join(rows) + "\n")


def test_sequence_discovery_excludes_ignore_and_whole_files(tmp_path):
    _write_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    _write_ignore(tmp_path, "M0101", ["1,9,100,100,50,50,1,-1,-1"])
    (_gt_dir(tmp_path) / "M0101_gt_whole.txt").write_text("1,1,0,0,10,10,1,1,3\n")

    dataset = load_uavdt(root=tmp_path, split="all")

    assert [s.name for s in dataset.sequences] == ["M0101"]


def test_seq_length_derived_from_last_annotated_frame(tmp_path):
    _write_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1", "3,1,2,0,10,10,1,1,-1"])

    dataset = load_uavdt(root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert seq.num_timesteps == 3


def test_unknown_split_raises(tmp_path):
    _write_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])

    with pytest.raises(ValueError, match="unknown uavdt split"):
        load_uavdt(root=tmp_path, split="test")


def test_missing_gt_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="split directory not found"):
        load_uavdt(root=tmp_path, split="all")


def test_malformed_gt_row_raises(tmp_path):
    _write_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1", "notaframe,1,1"])

    with pytest.raises(ValueError, match="malformed MOT row"):
        load_uavdt(root=tmp_path, split="all")


def test_malformed_ignore_row_raises(tmp_path):
    _write_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    _write_ignore(tmp_path, "M0101", ["notaframe,9,100,100,50,50,1,-1,-1"])

    with pytest.raises(ValueError, match="malformed MOT row"):
        load_uavdt(root=tmp_path, split="all")


def test_ignore_file_parses_into_gt_sequence_ignore_regions(tmp_path):
    _write_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    _write_ignore(tmp_path, "M0101", ["1,9,100,100,50,50,1,-1,-1"])

    dataset = load_uavdt(root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert len(seq.ignore_regions) == 1
    region = seq.ignore_regions[0]
    assert (region.frame, region.x, region.y, region.w, region.h) == (1, 100.0, 100.0, 50.0, 50.0)


def test_missing_ignore_file_yields_no_ignore_regions(tmp_path):
    # Legacy layout has no marker distinguishing "deliberately no ignore
    # regions" from "file just doesn't exist" -- absence means no regions.
    _write_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])

    dataset = load_uavdt(root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert seq.ignore_regions == ()


def test_gt_class_id_explicitly_stamped_from_config(tmp_path):
    # Non-tautological: read_mot defaults every row to class_id=1, so this only
    # passes if `_load_sequence` actually stamps `config.class_id` rather than
    # passing rows through untouched.
    _write_gt(tmp_path, "M0101", ["1,1,0,0,10,10,1,1,-1"])
    config = replace(
        UAVDT_CONFIG,
        default_root=tmp_path,
        class_id=7,
        protocol=replace(UAVDT_PROTOCOL, eval_classes=(7,)),
    )

    dataset = load_motchallenge(config, root=tmp_path, split="all")

    (seq,) = dataset.sequences
    assert all(track.class_id == 7 for track in seq.tracks)


def test_end_to_end_evaluate_with_independently_numbered_predictions(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    _write_gt(gt_root, "M0101", ["1,1,0,0,10,10,1,1,-1", "2,1,1,0,10,10,1,1,-1"])

    dataset = load_uavdt(root=gt_root, split="all")
    (seq,) = dataset.sequences
    # Predictions numbered independently of GT: different track id, disjoint
    # single-frame authored box (not a GT copy).
    write_mot(
        pred_dir / f"{seq.name}.txt",
        [Track(frame=1, track_id=901, x=1, y=1, w=10, h=10, conf=0.9)],
    )

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.combined["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }


def test_prediction_inside_ignore_region_excluded_through_evaluate(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    _write_gt(gt_root, "M0101", ["1,1,0,0,10,10,1,1,-1", "2,1,1,0,10,10,1,1,-1"])
    _write_ignore(gt_root, "M0101", ["1,9,100,100,50,50,1,-1,-1", "2,9,100,100,50,50,1,-1,-1"])

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


def test_ignore_regions_change_evaluation_results(tmp_path):
    # Control: identical GT and predictions, with vs. without the ignore file,
    # must produce different Count Dets -- proves the regions have effect.
    def _build_gt(root: Path, with_ignore: bool) -> None:
        _write_gt(root, "M0101", ["1,1,0,0,10,10,1,1,-1", "2,1,1,0,10,10,1,1,-1"])
        if with_ignore:
            _write_ignore(root, "M0101", ["1,9,100,100,50,50,1,-1,-1", "2,9,100,100,50,50,1,-1,-1"])

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
