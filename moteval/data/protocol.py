from dataclasses import dataclass

from moteval.data.model import FrameConvention


@dataclass(frozen=True)
class Protocol:
    """``distractor_classes`` are matched against but never scored. ``matching_fill`` is the
    score below-threshold pairs get in the distractor matching: TrackEval uses ``0`` on its
    box path and ``-10000`` on its MOTS path.
    """

    name: str
    frame_convention: FrameConvention
    eval_classes: tuple[int, ...]
    distractor_classes: tuple[int, ...] = ()
    distractor_iou_threshold: float = 0.5
    ignore_iou_threshold: float = 0.5
    drop_zero_conf_gt: bool = True
    matching_fill: float = 0.0
