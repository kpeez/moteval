import dataclasses

import numpy as np
import pytest

from moteval.data.convert import build_sequence_data
from moteval.data.model import FrameConvention, GtSequence
from moteval.formats.mot_txt import Track


def _seq() -> GtSequence:
    tracks = (
        Track(frame=1, track_id=7, x=0, y=0, w=10, h=10, conf=1.0),
        Track(frame=2, track_id=7, x=1, y=0, w=10, h=10, conf=1.0),
        Track(frame=1, track_id=42, x=50, y=50, w=10, h=10, conf=1.0),
    )
    return GtSequence(name="s", num_timesteps=2, tracks=tracks)


def test_sequence_data_is_frozen():
    data = build_sequence_data(_seq(), (), FrameConvention("1-indexed", 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(data, "name", "mutated")  # noqa: B010  (assignment form fails static checks)


def test_ids_are_densified_via_sorted_mapping():
    # raw ids 7 and 42 densify to 0 and 1, never a max-id-sized dense array.
    data = build_sequence_data(_seq(), (), FrameConvention("1-indexed", 1))
    assert data.num_gt_ids == 2
    np.testing.assert_array_equal(np.sort(data.gt_ids[0]), [0, 1])
    np.testing.assert_array_equal(data.gt_ids[1], [0])


def test_counts_are_summed_over_frames():
    data = build_sequence_data(_seq(), (), FrameConvention("1-indexed", 1))
    assert data.num_gt_dets == 3
    assert data.num_pred_dets == 0
