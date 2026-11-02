from collections.abc import Callable
from pathlib import Path
from typing import Any

from moteval import _moteval
from moteval._rust import call
from moteval.data.model import FrameConvention, GtSequence, MaskGtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats import MaskTrack, Track

DatasetLoader = Callable[..., MOTDataset[Any]]


def _dataset(raw: Any) -> MOTDataset[Any]:
    name, split, protocol, sequences = raw
    (
        protocol_name,
        (convention_name, first_frame),
        eval_classes,
        distractor_classes,
        distractor_iou_threshold,
        ignore_iou_threshold,
        drop_zero_conf_gt,
        matching_fill,
    ) = protocol
    built: list[GtSequence | MaskGtSequence] = []
    for kind, seq_name, num_timesteps, tracks, ignore_regions in sequences:
        if kind == "masks":
            built.append(
                MaskGtSequence(
                    name=seq_name,
                    num_timesteps=num_timesteps,
                    tracks=tuple(MaskTrack(*row) for row in tracks),
                    ignore_regions=tuple(MaskTrack(*row) for row in ignore_regions),
                )
            )
        else:
            built.append(
                GtSequence(
                    name=seq_name,
                    num_timesteps=num_timesteps,
                    tracks=tuple(Track(*row) for row in tracks),
                    ignore_regions=tuple(Track(*row) for row in ignore_regions),
                )
            )
    return MOTDataset(
        name=name,
        split=split,
        sequences=tuple(built),
        protocol=Protocol(
            name=protocol_name,
            frame_convention=FrameConvention(name=convention_name, first_frame=first_frame),
            eval_classes=tuple(eval_classes),
            distractor_classes=tuple(distractor_classes),
            distractor_iou_threshold=distractor_iou_threshold,
            ignore_iou_threshold=ignore_iou_threshold,
            drop_zero_conf_gt=drop_zero_conf_gt,
            matching_fill=matching_fill,
        ),
    )


def _loader(name: str, default_split: str) -> DatasetLoader:
    def load(root: str | Path, split: str = default_split) -> MOTDataset[Any]:
        return _dataset(call(_moteval.load_benchmark, name, root, split))

    load.__name__ = load.__qualname__ = f"load_{name}"
    load.__doc__ = f"Load the {name} ground truth under ``root`` (default split {default_split!r})."
    return load


BENCHMARKS: dict[str, DatasetLoader] = {
    name: _loader(name, split) for name, split in _moteval.benchmarks_table()
}


def default_data_root() -> Path:
    return call(_moteval.default_data_root)


def load_dataset(
    name: str, root: str | Path | None = None, split: str | None = None
) -> MOTDataset[GtSequence | MaskGtSequence]:
    """``root`` defaults to ``default_data_root() / name``; ``split`` to the benchmark's own."""
    return _dataset(call(_moteval.load_benchmark, name, root, split))


def load_motchallenge(root: str | Path, split: str = "train") -> MOTDataset[GtSequence]:
    """Read ``<root>/<split>/<seq>/gt/gt.txt`` and ``seqinfo.ini``: 1-indexed frames, class 1
    scored, no distractors, GT rows with a zero "consider" flag dropped. The dataset is
    named after ``root``.
    """
    return _dataset(call(_moteval.load_motchallenge, root, split))


def load_mots(
    root: str | Path, split: str = "train", class_id: int = 2
) -> MOTDataset[MaskGtSequence]:
    """Read ``<root>/<split>/<seq>/gt/gt.txt`` (MOTS rows) and ``seqinfo.ini``: 1-indexed
    frames, ``class_id`` scored (2 is the MOTS pedestrian class), class-10 rows as ignore
    regions. The dataset is named after ``root``.
    """
    return _dataset(call(_moteval.load_mots, root, split, class_id))
