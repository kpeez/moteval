"""Regression proof for the frame-indexing contract (issue #4).

Historically track-zoo silently dropped frame 0 of 0-indexed predictions evaluated
under a 1-indexed assumption (bit ChimpACT). These tests prove that failure mode is
structurally impossible: out-of-range frames raise loudly through the real
`evaluate()` path, and a correctly declared 0-indexed benchmark evaluates
identically to the same data re-encoded 1-indexed. Every prediction file below is
numbered independently of the ground truth -- never derived from GT frame lists.
"""

import pytest

from moteval import evaluate, load_dataset
from moteval.data.model import FrameConvention, GtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats.mot_txt import Track, write_mot
from moteval.metrics.count import Count


def test_zero_indexed_predictions_against_one_indexed_benchmark_raise_on_frame_0(tmp_path):
    dataset = load_dataset("toy")
    seq = dataset.sequences[0]
    pred_tracks = [
        Track(frame=f, track_id=999, x=0, y=0, w=10, h=10, conf=1.0) for f in range(0, 5)
    ]
    write_mot(tmp_path / f"{seq.name}.txt", pred_tracks)

    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count()])

    message = str(exc.value)
    assert "0" in message
    assert "1-indexed" in message


def test_predictions_beyond_sequence_length_raise_loud_error(tmp_path):
    dataset = load_dataset("toy")
    seq = dataset.sequences[0]
    pred_tracks = [Track(frame=6, track_id=999, x=0, y=0, w=10, h=10, conf=1.0)]
    write_mot(tmp_path / f"{seq.name}.txt", pred_tracks)

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
        for f in range(0, 5)
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
    # Frame index 0 (0-indexed frame 0 / 1-indexed frame 1) is never silently
    # dropped: all 5 frames' 2 predictions each are counted on both sides.
    assert zero_result.combined["Count"]["Dets"] == 10.0
