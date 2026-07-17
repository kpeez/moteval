import pytest

from moteval import evaluate, load_dataset
from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention, GtSequence
from moteval.formats.mot_txt import Track, write_mot
from moteval.metrics.count import Count


def test_out_of_range_pred_frame_raises_naming_frame_and_convention():
    gt = GtSequence(name="s", num_timesteps=5, tracks=())
    pred = (Track(frame=6, track_id=1, x=0, y=0, w=10, h=10, conf=1.0),)
    with pytest.raises(ValueError) as exc:
        build_sequence_data(gt, pred, FrameConvention("1-indexed", 1))
    message = str(exc.value)
    assert "6" in message
    assert "1-indexed" in message


def test_zero_indexed_pred_against_one_indexed_benchmark_raises(tmp_path):
    # Predictions numbered independently of GT: a frame-0 row is invalid here.
    dataset = load_dataset("toy")
    seq = dataset.sequences[0]
    write_mot(
        tmp_path / f"{seq.name}.txt",
        [Track(frame=0, track_id=1, x=0, y=0, w=10, h=10, conf=1.0)],
    )
    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count()])
    assert "1-indexed" in str(exc.value)
