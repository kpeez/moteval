from pathlib import Path

import pytest

from moteval import evaluate
from moteval.benchmarks.dancetrack import load_dancetrack
from moteval.benchmarks.motchallenge import MOTChallengeConfig, load_motchallenge
from moteval.benchmarks.sportsmot import load_sportsmot
from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import write_mot
from moteval.metrics.count import Count

_CONVENTION = FrameConvention(name="1-indexed", first_frame=1)
_TEST_PROTOCOL = Protocol(name="test-motchallenge", frame_convention=_CONVENTION, eval_classes=(1,))


def _test_config(root: Path) -> MOTChallengeConfig:
    return MOTChallengeConfig(name="test-motchallenge", default_root=root, protocol=_TEST_PROTOCOL)


def _write_seqinfo(seq_dir: Path, seq_length: int | str) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq_dir.name}\nimDir=img1\nframeRate=20\n"
        f"seqLength={seq_length}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )


def _write_gt(seq_dir: Path, rows: list[str]) -> None:
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "gt.txt").write_text("\n".join(rows) + "\n" if rows else "")


def _make_sequence(
    root: Path, split: str, seq_name: str, seq_length: int | str, rows: list[str]
) -> Path:
    seq_dir = root / split / seq_name
    _write_seqinfo(seq_dir, seq_length)
    _write_gt(seq_dir, rows)
    return seq_dir


def test_sequence_discovery_sorted_by_name(tmp_path):
    for name in ("seq-02", "seq-10", "seq-01"):
        _make_sequence(tmp_path, "val", name, 1, ["1,1,10,10,20,20,1,1,1"])

    dataset = load_motchallenge(_test_config(tmp_path), split="val")

    assert [seq.name for seq in dataset.sequences] == ["seq-01", "seq-02", "seq-10"]


def test_seq_length_comes_from_seqinfo_not_max_gt_frame(tmp_path):
    _make_sequence(
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

    dataset = load_motchallenge(_test_config(tmp_path), split="val")

    (seq,) = dataset.sequences
    assert seq.num_timesteps == 10


def test_missing_seqinfo_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    _write_gt(seq_dir, ["1,1,10,10,20,20,1,1,1"])

    with pytest.raises(ValueError, match="seqinfo.ini"):
        load_motchallenge(_test_config(tmp_path), split="val")


def test_seqinfo_missing_seqlength_key_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    seq_dir.mkdir(parents=True)
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=seq-01\nframeRate=20\n")
    _write_gt(seq_dir, ["1,1,10,10,20,20,1,1,1"])

    with pytest.raises(ValueError, match="seqLength"):
        load_motchallenge(_test_config(tmp_path), split="val")


def test_seqinfo_non_integer_seqlength_raises(tmp_path):
    _make_sequence(
        tmp_path, "val", "seq-01", seq_length="not-a-number", rows=["1,1,10,10,20,20,1,1,1"]
    )

    with pytest.raises(ValueError, match="seqLength"):
        load_motchallenge(_test_config(tmp_path), split="val")


def test_missing_gt_txt_raises(tmp_path):
    seq_dir = tmp_path / "val" / "seq-01"
    _write_seqinfo(seq_dir, 5)

    with pytest.raises(ValueError, match="gt.txt"):
        load_motchallenge(_test_config(tmp_path), split="val")


def test_malformed_gt_row_raises_and_names_file(tmp_path):
    seq_dir = _make_sequence(
        tmp_path, "val", "seq-01", seq_length=2, rows=["1,1,10,10,20,20,1,1,1", "notaframe,1,1"]
    )

    with pytest.raises(ValueError) as exc:
        load_motchallenge(_test_config(tmp_path), split="val")
    assert str(seq_dir / "gt" / "gt.txt") in str(exc.value)


def test_gt_class_id_defaults_to_pedestrian(tmp_path):
    # Regression for the read_mot hazard (issue #10 comment): read_mot parses no
    # class column, so every loaded GT Track silently lands on class_id=1. For
    # single-class MOTChallenge benchmarks that's coincidentally correct.
    _make_sequence(
        tmp_path,
        "val",
        "seq-01",
        seq_length=2,
        rows=["1,1,10,10,20,20,1,1,1", "2,1,12,10,20,20,1,1,1"],
    )

    dataset = load_motchallenge(_test_config(tmp_path), split="val")

    (seq,) = dataset.sequences
    assert all(track.class_id == 1 for track in seq.tracks)


def test_id_densification_through_build_sequence_data(tmp_path):
    _make_sequence(
        tmp_path,
        "val",
        "seq-01",
        seq_length=2,
        rows=[
            "1,5,10,10,20,20,1,1,1",
            "1,42,50,50,20,20,1,1,1",
            "2,5,12,10,20,20,1,1,1",
            "2,42,52,50,20,20,1,1,1",
        ],
    )

    dataset = load_motchallenge(_test_config(tmp_path), split="val")
    (seq,) = dataset.sequences
    data = build_sequence_data(seq, (), _TEST_PROTOCOL, 1)

    assert data.num_gt_ids == 2
    for frame_ids in data.gt_ids:
        assert set(frame_ids.tolist()) <= {0, 1}


def test_frames_binned_regardless_of_file_order(tmp_path):
    _make_sequence(
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

    dataset = load_motchallenge(_test_config(tmp_path), split="val")
    (seq,) = dataset.sequences
    data = build_sequence_data(seq, (), _TEST_PROTOCOL, 1)

    assert data.num_timesteps == 3
    assert [frame.shape[0] for frame in data.gt_ids] == [1, 1, 1]


def test_root_override_does_not_touch_default_root(tmp_path):
    _make_sequence(tmp_path, "val", "dancetrack0001", seq_length=2, rows=["1,1,10,10,20,20,1,1,1"])

    dataset = load_dancetrack(root=tmp_path, split="val")

    assert dataset.name == "dancetrack"
    assert [seq.name for seq in dataset.sequences] == ["dancetrack0001"]


def test_sportsmot_loads_with_explicit_root(tmp_path):
    _make_sequence(tmp_path, "val", "v_00001", seq_length=2, rows=["1,1,10,10,20,20,1,1,1"])

    dataset = load_sportsmot(root=tmp_path, split="val")

    assert dataset.name == "sportsmot"
    assert [seq.name for seq in dataset.sequences] == ["v_00001"]


def test_end_to_end_evaluate_dancetrack_layout_fixture(tmp_path):
    gt_root = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()

    _make_sequence(
        gt_root,
        "val",
        "dancetrack0001",
        seq_length=2,
        rows=[
            "1,1,10,10,20,20,1,1,1",
            "1,2,100,100,30,40,1,1,1",
            "2,1,12,10,20,20,1,1,1",
            "2,2,102,100,30,40,1,1,1",
        ],
    )
    _make_sequence(
        gt_root,
        "val",
        "dancetrack0002",
        seq_length=2,
        rows=[
            "1,1,50,50,25,25,1,1,1",
            "2,1,55,50,25,25,1,1,1",
        ],
    )

    dataset = load_dancetrack(root=gt_root, split="val")
    for seq in dataset.sequences:
        write_mot(pred_dir / f"{seq.name}.txt", list(seq.tracks))

    result = evaluate(dataset, pred_dir, [Count()])

    assert result.per_sequence["dancetrack0001"]["Count"] == {
        "Dets": 4.0,
        "GT_Dets": 4.0,
        "IDs": 2.0,
        "GT_IDs": 2.0,
    }
    assert result.per_sequence["dancetrack0002"]["Count"] == {
        "Dets": 2.0,
        "GT_Dets": 2.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }
    assert result.combined["Count"] == {
        "Dets": 6.0,
        "GT_Dets": 6.0,
        "IDs": 3.0,
        "GT_IDs": 3.0,
    }
