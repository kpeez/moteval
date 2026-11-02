from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

import numpy as np

from moteval.formats import MaskTrack, Track

if TYPE_CHECKING:
    from moteval.data.protocol import Protocol


class RleMask(TypedDict):
    """``counts`` admits ``str`` to match ``_EncodedRLE`` in typeshed's pycocotools stub,
    whose TypedDict fields are invariant.
    """

    size: list[int]
    counts: str | bytes


@dataclass(frozen=True)
class FrameConvention:
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
    """``ignore_regions`` are crowd-ignore boxes as `Track` rows, whose id, conf and class
    are irrelevant.
    """

    name: str
    num_timesteps: int
    tracks: tuple[Track, ...]
    ignore_regions: tuple[Track, ...] = ()


@dataclass(frozen=True)
class MaskGtSequence:
    """``ignore_regions`` are the ignore masks (class-10 rows of a MOTS GT file)."""

    name: str
    num_timesteps: int
    tracks: tuple[MaskTrack, ...]
    ignore_regions: tuple[MaskTrack, ...] = ()


SeqT_co = TypeVar("SeqT_co", bound="GtSequence | MaskGtSequence", covariant=True)


@dataclass(frozen=True)
class MOTDataset(Generic[SeqT_co]):
    name: str
    split: str
    sequences: tuple[SeqT_co, ...]
    protocol: "Protocol"


@dataclass(frozen=True)
class BoxGeometry:
    gt: tuple[np.ndarray, ...]
    pred: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class MaskGeometry:
    gt: tuple[np.ndarray, ...]
    pred: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SequenceData:
    """Every per-frame field is a tuple of length ``num_timesteps``; ids are densified to
    ``0..num_*_ids-1``. ``similarity`` holds each frame's GT x prediction IoU.
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
