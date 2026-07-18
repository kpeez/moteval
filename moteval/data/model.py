"""Canonical typed data model: raw GT sequences and frame-major `SequenceData`.

All metrics compute on `SequenceData` alone: a frozen, frame-major container of
per-frame ragged arrays (densified ids, confidences, xywh boxes) plus precomputed
per-frame box-IoU similarity matrices.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from moteval.formats.mot_txt import Track

if TYPE_CHECKING:
    from moteval.data.protocol import Protocol


@dataclass(frozen=True)
class FrameConvention:
    """Declared frame-indexing convention for a benchmark.

    ``first_frame`` is the frame number of the first timestep (1 for MOTChallenge,
    0 for natively 0-indexed benchmarks). Frame numbers must fall in
    ``[first_frame, first_frame + num_timesteps)``; anything else raises loudly.
    """

    name: str
    first_frame: int

    def to_index(self, frame: int, num_timesteps: int) -> int:
        index = frame - self.first_frame
        if index < 0 or index >= num_timesteps:
            valid = range(self.first_frame, self.first_frame + num_timesteps)
            raise ValueError(
                f"frame {frame} is out of range for convention {self.name!r} "
                f"(first_frame={self.first_frame}); valid frames are "
                f"[{valid.start}, {valid.stop - 1}]"
            )
        return index


@dataclass(frozen=True)
class GtSequence:
    """Ground truth for one sequence, before predictions are merged in.

    ``ignore_regions`` are per-frame crowd-ignore boxes (as `Track` rows, whose id,
    conf and class are irrelevant); predictions falling inside them are dropped by
    the preprocessing engine per the sequence's `Protocol`.
    """

    name: str
    num_timesteps: int
    tracks: tuple[Track, ...]
    ignore_regions: tuple[Track, ...] = ()


@dataclass(frozen=True)
class MOTDataset:
    """A named, split-scoped collection of ground-truth sequences.

    ``protocol`` declares the benchmark's preprocessing (classes, distractors,
    ignore-region IoA threshold, conf-zero semantics) and its frame convention.
    """

    name: str
    split: str
    sequences: tuple[GtSequence, ...]
    protocol: "Protocol"


@dataclass(frozen=True)
class SequenceData:
    """Frozen frame-major evaluation input for one sequence.

    Every per-frame field is a tuple of length ``num_timesteps``. Ids are densified
    to ``0..num_*_ids-1`` via a dict mapping over sorted unique raw ids.
    """

    name: str
    num_timesteps: int
    num_gt_ids: int
    num_pred_ids: int
    num_gt_dets: int
    num_pred_dets: int
    gt_ids: tuple[np.ndarray, ...]
    pred_ids: tuple[np.ndarray, ...]
    pred_confidences: tuple[np.ndarray, ...]
    gt_boxes: tuple[np.ndarray, ...]
    pred_boxes: tuple[np.ndarray, ...]
    similarity: tuple[np.ndarray, ...] = field(repr=False)
