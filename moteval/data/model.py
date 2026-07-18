"""Canonical typed data model: raw GT sequences and frame-major `SequenceData`.

All metrics compute on `SequenceData` alone: a frozen, frame-major container of
per-frame ragged arrays (densified ids, confidences, boxes-or-masks geometry)
plus precomputed per-frame similarity matrices (box IoU or mask IoU).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from moteval.formats.mot_txt import Track
from moteval.formats.mots_txt import MaskTrack

if TYPE_CHECKING:
    from moteval.data.protocol import Protocol

# pycocotools compressed RLE: {"size": [height, width], "counts": bytes}.
RleMask = dict[str, object]


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
class MaskGtSequence:
    """Ground truth for one mask (MOTS) sequence, before predictions are merged in.

    ``ignore_regions`` are per-frame ignore masks (class-10 rows in MOTS GT files,
    routed here by the loader); unmatched predictions overlapping their per-frame
    union are dropped by the preprocessing engine per the sequence's `Protocol`.
    """

    name: str
    num_timesteps: int
    tracks: tuple[MaskTrack, ...]
    ignore_regions: tuple[MaskTrack, ...] = ()


SeqT_co = TypeVar("SeqT_co", bound="GtSequence | MaskGtSequence", covariant=True)


@dataclass(frozen=True)
class MOTDataset(Generic[SeqT_co]):
    """A named, split-scoped collection of ground-truth sequences.

    Generic over the sequence kind so box loaders return
    ``MOTDataset[GtSequence]`` and mask loaders ``MOTDataset[MaskGtSequence]``
    (covariant: either is a ``MOTDataset[GtSequence | MaskGtSequence]``).
    ``protocol`` declares the benchmark's preprocessing (classes, distractors,
    ignore-region IoA threshold, conf-zero semantics) and its frame convention.
    """

    name: str
    split: str
    sequences: tuple[SeqT_co, ...]
    protocol: "Protocol"


@dataclass(frozen=True)
class BoxGeometry:
    """Per-frame surviving xywh boxes: each frame is an ``(n, 4)`` float array."""

    gt: tuple[np.ndarray, ...]
    pred: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class MaskGeometry:
    """Per-frame surviving pycocotools-RLE masks: each frame is an object array
    of RLE dicts (kept as arrays so J&F can index them like the box arrays)."""

    gt: tuple[np.ndarray, ...]
    pred: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SequenceData:
    """Frozen frame-major evaluation input for one sequence.

    Every per-frame field is a tuple of length ``num_timesteps``. Ids are densified
    to ``0..num_*_ids-1`` via a dict mapping over sorted unique raw ids.
    ``geometry`` is the boxes-or-masks union; similarity is precomputed (box IoU
    or mask IoU) so metrics stay geometry-agnostic unless they opt into geometry
    (TrackMAP reads boxes, J&F reads masks) via the asserting accessors below.
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
    geometry: BoxGeometry | MaskGeometry
    similarity: tuple[np.ndarray, ...] = field(repr=False)

    @property
    def gt_boxes(self) -> tuple[np.ndarray, ...]:
        if not isinstance(self.geometry, BoxGeometry):
            raise TypeError(f"sequence {self.name!r} carries mask geometry, not boxes")
        return self.geometry.gt

    @property
    def pred_boxes(self) -> tuple[np.ndarray, ...]:
        if not isinstance(self.geometry, BoxGeometry):
            raise TypeError(f"sequence {self.name!r} carries mask geometry, not boxes")
        return self.geometry.pred

    @property
    def gt_masks(self) -> tuple[np.ndarray, ...]:
        if not isinstance(self.geometry, MaskGeometry):
            raise TypeError(f"sequence {self.name!r} carries box geometry, not masks")
        return self.geometry.gt

    @property
    def pred_masks(self) -> tuple[np.ndarray, ...]:
        if not isinstance(self.geometry, MaskGeometry):
            raise TypeError(f"sequence {self.name!r} carries box geometry, not masks")
        return self.geometry.pred
