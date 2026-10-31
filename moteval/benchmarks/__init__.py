"""Built-in benchmark loaders, keyed by name.

`BENCHMARKS` is the explicit index of built-in benchmarks (name -> loader); `load_dataset`
is the public by-name entry point; it reads ``default_data_root() / name`` unless the
caller passes ``root``. Custom data never registers anything: standard-layout
directories load through `load_motchallenge` / `load_mots`, and any other source format
just constructs a `MOTDataset` directly and calls `evaluate`.

Every loader runs in the Rust extension (``rust/moteval/src/benchmarks/``, one module per
benchmark); this module turns its plain data into the frozen dataclasses. Each loader
takes a required ``root`` and a ``split`` with the benchmark's default.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from moteval import _moteval
from moteval._rust import call
from moteval.data.model import FrameConvention, GtSequence, MaskGtSequence, MOTDataset
from moteval.data.protocol import Protocol
from moteval.formats import MaskTrack, Track

# Every loader takes a required root and a split keyword with a benchmark default; the
# sequences are `GtSequence` (box benchmarks) or `MaskGtSequence` (mots20).
DatasetLoader = Callable[..., MOTDataset[Any]]


def _dataset(raw: Any) -> MOTDataset[Any]:
    """A `MOTDataset` from the extension's ``(name, split, protocol, sequences)`` tuple."""
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
    """Return the directory that holds one subdirectory per built-in benchmark.

    ``MOTEVAL_DATA_ROOT`` wins when set; otherwise ``data/benchmarks``, relative to the
    working directory. This is also where ``scripts/download_benchmarks.py`` puts data.
    """
    return call(_moteval.default_data_root)


def load_dataset(
    name: str, root: str | Path | None = None, split: str | None = None
) -> MOTDataset[GtSequence | MaskGtSequence]:
    """Load a built-in benchmark by name, optionally overriding root and split.

    ``root`` defaults to ``default_data_root() / name``; ``split`` defaults to the
    benchmark's own default split.
    """
    return _dataset(call(_moteval.load_benchmark, name, root, split))


def load_motchallenge(root: str | Path, split: str = "train") -> MOTDataset[GtSequence]:
    """Load any standard MOTChallenge-layout directory, no registration required.

    For custom box data at ``<root>/<split>/<seq>/gt/gt.txt`` + ``seqinfo.ini``:
    1-indexed frames, single evaluated class (1), no distractors, conf-zero GT
    rows dropped (the gt.txt "consider" flag). The dataset is named after the
    root directory. Benchmarks with their own quirks get a dedicated loader
    instead; this is the seam for data that already matches the plain layout.
    """
    return _dataset(call(_moteval.load_motchallenge, root, split))


def load_mots(
    root: str | Path, split: str = "train", class_id: int = 2
) -> MOTDataset[MaskGtSequence]:
    """Load any standard MOTS-layout directory, no registration required.

    For custom mask data at ``<root>/<split>/<seq>/gt/gt.txt`` (whitespace MOTS
    rows) + ``seqinfo.ini``: 1-indexed frames, class-10 rows routed to ignore
    regions, upstream MOTS matching semantics (``matching_fill=-10000``, GT never
    conf-filtered). ``class_id`` selects the evaluated class (MOTS pedestrian
    convention is 2). The dataset is named after the root directory.
    """
    return _dataset(call(_moteval.load_mots, root, split, class_id))
