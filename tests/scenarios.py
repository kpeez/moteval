"""Declarative synthetic scenarios shared by the parity fixture suite.

These scenarios are the union of the fixtures from the old live-oracle parity tests
(tests/parity/test_oracle_{synthetic,clear,identity,jf,trackmap}.py). They are consumed
by two callers:

- ``scripts/regen_parity_fixtures.py`` runs official TrackEval (pinned commit) on them
  and freezes the results as JSON under ``tests/fixtures/``.
- The parity test suite runs moteval on the same scenarios and asserts exact equality
  against those frozen fixtures.

Box/MOTS scenarios are written to disk in the SKIP_SPLIT_FOL MOTChallenge layout
(``gt/{seq}/gt/gt.txt`` + ``trackers/oracle/data/{seq}.txt``) so both sides read the
identical files. Predictions are numbered independently of GT (disjoint id ranges).
Every box GT row sets ``zero_marked=1`` and the fixtures are scored with
``do_preproc=False``, so both sides see the identical, unfiltered detections.

TrackMAP is not wired into the MOTChallenge dataset upstream, so its scenarios stay
pure Python specs (``{track_id: {frame: box}}`` for GT, ``{track_id: {frame: (box,
confidence)}}`` for predictions); `build_trackmap_sequence_data` builds moteval's real
input from a spec, and the regen script builds the upstream-shaped ``data`` dict from
the same spec.

Real-data cases (DanceTrack val, SportsMOT val, one MOTS20 sequence) generate seeded
perturbed predictions via `tests.perturb` into a caller-provided tmp dir, mirroring the
old tests/parity/test_real_data.py setup exactly.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from moteval.benchmarks import default_data_root
from moteval.benchmarks.dancetrack import load_dancetrack
from moteval.benchmarks.mots20 import MOTS20_IGNORE_CLASS, MOTS20_PROTOCOL, load_mots20
from moteval.benchmarks.sportsmot import load_sportsmot
from moteval.data.model import (
    BoxGeometry,
    FrameConvention,
    GtSequence,
    MaskGtSequence,
    MOTDataset,
    SequenceData,
)
from moteval.data.protocol import Protocol
from moteval.data.similarity import box_iou
from moteval.formats import MaskTrack, read_mot, read_mots, write_mot, write_mots
from tests.perturb import perturb_box_tracks, perturb_mask_tracks
from tests.rle import encode_mask

BOX_PROTOCOL = Protocol(
    name="parity",
    frame_convention=FrameConvention(name="1-indexed", first_frame=1),
    eval_classes=(1,),
)
BOX_METRICS = ("HOTA", "CLEAR", "Identity", "Count")
COMBINE_CLASSES_SCENARIO = "combine_classes"
# The MOTS scenario that the multi-class mask check reuses (see `class_views`).
MOTS_MULTI_CLASS_SCENARIO = "jf_perturbed_predictions"

REAL_DATA_SEED = 20260718


@dataclass(frozen=True)
class Scenario:
    """One named scenario: sequences of ``(seq_name, num_timesteps, gt_rows, pred_rows)``.

    Box rows are MOT-txt value lists; MOTS rows are `MaskTrack` instances.
    """

    name: str
    sequences: tuple[tuple[str, int, list, list], ...]


# ---------------------------------------------------------------------------
# Box scenarios (HOTA / CLEAR / Identity / Count)
# ---------------------------------------------------------------------------

_MISS_GT = [
    [1, 1, 10, 10, 20, 40, 1, 1, 1],
    [2, 1, 12, 10, 20, 40, 1, 1, 1],
    [3, 1, 14, 10, 20, 40, 1, 1, 1],
    [1, 2, 100, 50, 30, 60, 1, 1, 1],
    [2, 2, 102, 50, 30, 60, 1, 1, 1],
    [3, 2, 104, 50, 30, 60, 1, 1, 1],
]
_MISS_PRED = [
    [1, 101, 10, 10, 20, 40, 1],
    [2, 101, 12, 10, 20, 40, 1],
    [3, 101, 14, 10, 20, 40, 1],
]
_FP_GT = [
    [1, 1, 10, 10, 20, 40, 1, 1, 1],
    [2, 1, 12, 10, 20, 40, 1, 1, 1],
]
_FP_PRED = [
    [1, 201, 10, 10, 20, 40, 1],
    [2, 201, 12, 10, 20, 40, 1],
    [1, 202, 500, 500, 10, 10, 1],
    [2, 202, 500, 500, 10, 10, 1],
]
_GAP_GT = [
    [1, 1, 10, 10, 20, 40, 1, 1, 1],
    [1, 2, 500, 500, 20, 40, 1, 1, 1],
    [2, 1, 10, 10, 20, 40, 1, 1, 1],
    [2, 2, 500, 500, 20, 40, 1, 1, 1],
    [3, 2, 500, 500, 20, 40, 1, 1, 1],
    [4, 1, 10, 10, 20, 40, 1, 1, 1],
    [4, 2, 500, 500, 20, 40, 1, 1, 1],
]

BOX_SCENARIOS: tuple[Scenario, ...] = (
    # gt id 2 is never predicted at all: pure false negatives across every frame.
    Scenario("misses", (("MISS01", 3, _MISS_GT, _MISS_PRED),)),
    # every gt det matches; extra pred dets have no gt counterpart at all.
    Scenario("false_positives", (("FP01", 2, _FP_GT, _FP_PRED),)),
    # one continuous gt track, fully detected every frame, but the predicted id
    # changes partway through: full DetA, degraded AssA, one CLEAR IDSW.
    Scenario(
        "id_switches",
        (
            (
                "IDSW01",
                4,
                [
                    [1, 1, 10, 10, 20, 40, 1, 1, 1],
                    [2, 1, 12, 10, 20, 40, 1, 1, 1],
                    [3, 1, 14, 10, 20, 40, 1, 1, 1],
                    [4, 1, 16, 10, 20, 40, 1, 1, 1],
                ],
                [
                    [1, 301, 10, 10, 20, 40, 1],
                    [2, 301, 12, 10, 20, 40, 1],
                    [3, 302, 14, 10, 20, 40, 1],
                    [4, 302, 16, 10, 20, 40, 1],
                ],
            ),
        ),
    ),
    # frame counts vary independently on the gt and pred side per frame.
    Scenario(
        "ragged_frames",
        (
            (
                "RAGGED01",
                4,
                [
                    [1, 1, 0, 0, 10, 10, 1, 1, 1],
                    [2, 1, 0, 0, 10, 10, 1, 1, 1],
                    [2, 2, 50, 50, 10, 10, 1, 1, 1],
                    [2, 3, 90, 10, 10, 10, 1, 1, 1],
                    [4, 1, 0, 0, 10, 10, 1, 1, 1],
                    [4, 4, 200, 200, 10, 10, 1, 1, 1],
                ],
                [
                    [1, 401, 0, 0, 10, 10, 1],
                    [2, 401, 0, 0, 10, 10, 1],
                    [2, 402, 50, 50, 10, 10, 1],
                    [3, 403, 500, 500, 10, 10, 1],
                    [4, 401, 0, 0, 10, 10, 1],
                    [4, 404, 200, 200, 10, 10, 1],
                    [4, 405, 300, 300, 10, 10, 1],
                ],
            ),
        ),
    ),
    # frames 2 and 4 have no gt and no pred at all.
    Scenario(
        "empty_frames",
        (
            (
                "EMPTYF01",
                5,
                [
                    [1, 1, 10, 10, 20, 40, 1, 1, 1],
                    [3, 1, 14, 10, 20, 40, 1, 1, 1],
                    [5, 1, 18, 10, 20, 40, 1, 1, 1],
                ],
                [
                    [1, 501, 10, 10, 20, 40, 1],
                    [3, 501, 14, 10, 20, 40, 1],
                    [5, 501, 18, 10, 20, 40, 1],
                ],
            ),
        ),
    ),
    # three gt boxes and three pred boxes stacked at identical coordinates: every
    # gt/pred pair has IoU 1.0, forcing scipy to break ties among equal-cost matches.
    Scenario(
        "tie_heavy_assignment",
        (
            (
                "TIE01",
                1,
                [
                    [1, 1, 10, 10, 20, 20, 1, 1, 1],
                    [1, 2, 10, 10, 20, 20, 1, 1, 1],
                    [1, 3, 10, 10, 20, 20, 1, 1, 1],
                ],
                [
                    [1, 601, 10, 10, 20, 20, 1],
                    [1, 602, 10, 10, 20, 20, 1],
                    [1, 603, 10, 10, 20, 20, 1],
                ],
            ),
        ),
    ),
    Scenario(
        "multi_sequence_combine",
        (
            (
                "MISS01",
                2,
                [
                    [1, 1, 10, 10, 20, 40, 1, 1, 1],
                    [2, 1, 12, 10, 20, 40, 1, 1, 1],
                    [1, 2, 100, 50, 30, 60, 1, 1, 1],
                    [2, 2, 102, 50, 30, 60, 1, 1, 1],
                ],
                [
                    [1, 101, 10, 10, 20, 40, 1],
                    [2, 101, 12, 10, 20, 40, 1],
                ],
            ),
            ("FP01", 2, _FP_GT, _FP_PRED),
        ),
    ),
    # Two sequences with deliberately different det/FN counts. The class-combination
    # checks reuse them as two classes: the original HOTA-only
    # combine_classes_class_averaged entry, and the multi-class entries built from
    # `class_views` (see the regen script).
    Scenario(
        COMBINE_CLASSES_SCENARIO,
        (
            (
                "CLASSA01",
                2,
                [
                    [1, 1, 10, 10, 20, 40, 1, 1, 1],
                    [2, 1, 12, 10, 20, 40, 1, 1, 1],
                ],
                [
                    [1, 1001, 10, 10, 20, 40, 1],
                    [2, 1001, 12, 10, 20, 40, 1],
                ],
            ),
            (
                "CLASSB01",
                3,
                _MISS_GT,
                [
                    [1, 2001, 10, 10, 20, 40, 1],
                    [2, 2001, 12, 10, 20, 40, 1],
                    [3, 2001, 14, 10, 20, 40, 1],
                    [1, 2002, 500, 500, 10, 10, 1],
                ],
            ),
        ),
    ),
    Scenario("empty_gt", (("EMPTYGT01", 1, [], [[1, 1, 10, 10, 20, 40, 1]]),)),
    Scenario("empty_preds", (("EMPTYPRED01", 1, [[1, 1, 10, 10, 20, 40, 1, 1, 1]], []),)),
    Scenario("both_empty", (("EMPTYBOTH01", 1, [], []),)),
    # a gt track that exists for exactly one frame, matched cleanly.
    Scenario(
        "single_frame_track",
        (("SINGLE01", 1, [[1, 1, 10, 10, 20, 40, 1, 1, 1]], [[1, 701, 10, 10, 20, 40, 1]]),),
    ),
    # gt id 1 is absent from gt at frame 3 (gt id 2 anchors the frame so it stays
    # "considered"); the same pred id resumes after the gap -> Frag but no IDSW.
    # Exercises CLEAR's 1000x same-ID cost bonus keeping the resumed match.
    Scenario(
        "gap_then_resume_same_id",
        (
            (
                "GAP01",
                4,
                _GAP_GT,
                [
                    [1, 901, 10, 10, 20, 40, 1],
                    [1, 950, 500, 500, 20, 40, 1],
                    [2, 901, 10, 10, 20, 40, 1],
                    [2, 950, 500, 500, 20, 40, 1],
                    [3, 950, 500, 500, 20, 40, 1],
                    [4, 901, 10, 10, 20, 40, 1],
                    [4, 950, 500, 500, 20, 40, 1],
                ],
            ),
        ),
    ),
    # identical gap, but a different pred id resumes after it -> Frag and IDSW.
    Scenario(
        "gap_then_resume_different_id",
        (
            (
                "GAP02",
                4,
                _GAP_GT,
                [
                    [1, 901, 10, 10, 20, 40, 1],
                    [1, 950, 500, 500, 20, 40, 1],
                    [2, 901, 10, 10, 20, 40, 1],
                    [2, 950, 500, 500, 20, 40, 1],
                    [3, 950, 500, 500, 20, 40, 1],
                    [4, 902, 10, 10, 20, 40, 1],
                    [4, 950, 500, 500, 20, 40, 1],
                ],
            ),
        ),
    ),
    # Identity block-matrix shape asymmetry: 1 gt track, 3 pred tracks (2 pure FPs).
    Scenario(
        "more_pred_tracks_than_gt_tracks",
        (
            (
                "MOREPRED01",
                2,
                _FP_GT,
                [
                    [1, 801, 10, 10, 20, 40, 1],
                    [2, 801, 12, 10, 20, 40, 1],
                    [1, 802, 500, 500, 10, 10, 1],
                    [2, 803, 600, 600, 10, 10, 1],
                ],
            ),
        ),
    ),
    # Identity block-matrix shape asymmetry: 3 gt tracks, 1 pred track (2 pure FNs).
    Scenario(
        "more_gt_tracks_than_pred_tracks",
        (
            (
                "MOREGT01",
                2,
                [
                    [1, 1, 10, 10, 20, 40, 1, 1, 1],
                    [1, 2, 500, 500, 10, 10, 1, 1, 1],
                    [1, 3, 600, 600, 10, 10, 1, 1, 1],
                    [2, 1, 12, 10, 20, 40, 1, 1, 1],
                    [2, 2, 500, 500, 10, 10, 1, 1, 1],
                    [2, 3, 600, 600, 10, 10, 1, 1, 1],
                ],
                [
                    [1, 901, 10, 10, 20, 40, 1],
                    [2, 901, 12, 10, 20, 40, 1],
                ],
            ),
        ),
    ),
)


def _write_box_rows(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(",".join(str(v) for v in row) for row in rows)
    path.write_text(text + "\n" if rows else "")


def predictions_dir(tmp_dir: Path) -> Path:
    return tmp_dir / "trackers" / "oracle" / "data"


def write_mot_scenario(tmp_dir: Path, scenario: Scenario) -> dict[str, int]:
    """Write a box scenario in the SKIP_SPLIT_FOL layout; returns ``{seq: length}``."""
    seq_lengths = {}
    for name, num_timesteps, gt_rows, pred_rows in scenario.sequences:
        _write_box_rows(tmp_dir / "gt" / name / "gt" / "gt.txt", gt_rows)
        _write_box_rows(predictions_dir(tmp_dir) / f"{name}.txt", pred_rows)
        seq_lengths[name] = num_timesteps
    return seq_lengths


def build_box_dataset(tmp_dir: Path, scenario: Scenario) -> MOTDataset:
    """Build the moteval `MOTDataset` from the files `write_mot_scenario` wrote."""
    gt_sequences = tuple(
        GtSequence(
            name=name,
            num_timesteps=num_timesteps,
            tracks=tuple(read_mot(tmp_dir / "gt" / name / "gt" / "gt.txt")),
        )
        for name, num_timesteps, _, _ in scenario.sequences
    )
    return MOTDataset(name="synthetic", split="val", sequences=gt_sequences, protocol=BOX_PROTOCOL)


# ---------------------------------------------------------------------------
# Multi-class runs over existing scenarios
# ---------------------------------------------------------------------------
#
# The oracle's MOTChallenge and MOTS datasets score one class only. So a multi-class
# check takes an existing scenario plus a ``class_of(sequence name, row) -> class id``
# rule. moteval scores the scenario as one multi-class run, with each row labelled by
# that rule (`write_multi_class_*` + `build_multi_class_*`). The regen script scores
# each class through the oracle as one of the `class_views`: the same sequences with
# every other class's rows removed. That is exactly the per-class view TrackEval's
# evaluator combines before it combines classes.
#
# Boxes use combine_classes with `sequence_classes` (sequence ``i`` is class ``i + 1``).
# Masks cannot: a sequence without the class gives J&F NaN (upstream too), and the
# class combinations would then be NaN as well. So masks split one sequence by track id.

MULTI_CLASS_BOX_PROTOCOL = replace(BOX_PROTOCOL, name="parity-multi-class", eval_classes=(1, 2))
MULTI_CLASS_MOTS_PROTOCOL = replace(
    MOTS20_PROTOCOL, name="parity-mots-multi-class", eval_classes=(1, 2)
)
ClassOf = Callable[[str, Any], int]


def sequence_classes(scenario: Scenario) -> ClassOf:
    """Every row of sequence ``i`` is class ``i + 1``."""
    index = {name: i + 1 for i, (name, _, _, _) in enumerate(scenario.sequences)}
    return lambda sequence, _row: index[sequence]


def class_views(
    scenario: Scenario, class_of: ClassOf, classes: tuple[int, ...]
) -> tuple[Scenario, ...]:
    """One single-class scenario per class: only the rows ``class_of`` gives that class."""
    return tuple(
        Scenario(
            f"{scenario.name}-class-{cls}",
            tuple(
                (
                    name,
                    num_timesteps,
                    [row for row in gt_rows if class_of(name, row) == cls],
                    [row for row in pred_rows if class_of(name, row) == cls],
                )
                for name, num_timesteps, gt_rows, pred_rows in scenario.sequences
            ),
        )
        for cls in classes
    )


def write_multi_class_box_scenario(tmp_dir: Path, scenario: Scenario, class_of: ClassOf) -> None:
    """Write a box scenario with each row's class from ``class_of`` in column 8.

    GT rows already have a class column (index 7), which is replaced. Prediction rows
    are ``frame,id,x,y,w,h,conf``, so the class is appended as column 8.
    """
    for name, _, gt_rows, pred_rows in scenario.sequences:
        _write_box_rows(
            tmp_dir / "gt" / name / "gt" / "gt.txt",
            [[*row[:7], class_of(name, row), *row[8:]] for row in gt_rows],
        )
        _write_box_rows(
            predictions_dir(tmp_dir) / f"{name}.txt",
            [[*row, class_of(name, row)] for row in pred_rows],
        )


def build_multi_class_box_dataset(tmp_dir: Path, scenario: Scenario) -> MOTDataset:
    """Build the two-class `MOTDataset` from what `write_multi_class_box_scenario` wrote."""
    gt_sequences = tuple(
        GtSequence(
            name=name,
            num_timesteps=num_timesteps,
            tracks=tuple(read_mot(tmp_dir / "gt" / name / "gt" / "gt.txt", class_column=True)),
        )
        for name, num_timesteps, _, _ in scenario.sequences
    )
    return MOTDataset(
        name="synthetic-multi-class",
        split="val",
        sequences=gt_sequences,
        protocol=MULTI_CLASS_BOX_PROTOCOL,
    )


# ---------------------------------------------------------------------------
# MOTS scenarios (J&F plus mask HOTA/CLEAR/Identity/Count)
# ---------------------------------------------------------------------------

MASK_H, MASK_W = 48, 64


def _lane_mask(lane: int, col: int, width: int = 10) -> np.ndarray:
    """A 6-row-tall block in row lane ``lane`` starting at column ``col``.

    Lanes are 8 rows apart so masks in different lanes never overlap.
    """
    mask = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
    r0 = lane * 8
    c0 = max(0, min(col, MASK_W - width))
    mask[r0 : r0 + 6, c0 : c0 + width] = 1
    return mask


def _mask_row(frame: int, track_id: int, mask: np.ndarray, class_id: int = 2) -> MaskTrack:
    counts = encode_mask(mask)["counts"]
    assert isinstance(counts, bytes)
    return MaskTrack(
        frame=frame,
        track_id=track_id,
        class_id=class_id,
        img_h=MASK_H,
        img_w=MASK_W,
        rle=counts.decode(),
    )


def build_mots_scenarios() -> tuple[Scenario, ...]:
    """The MOTS scenarios ported from tests/parity/test_oracle_jf.py.

    Masks live in disjoint row lanes so no frame ever contains overlapping masks
    (the oracle rejects those). Built on demand because rows need RLE encoding.
    """
    # Three GT tracks over 8 frames; predictions are independently numbered and
    # perturbed: 101 covers GT 1 but skips frames 3-4 (padding path), GT 2 is covered
    # by 102 then 103 (ID switch), GT 3 is entirely missed, 104 is a pure drifting FP.
    frames = 8
    perturbed_gt: list[MaskTrack] = []
    perturbed_pred: list[MaskTrack] = []
    for f in range(1, frames + 1):
        perturbed_gt.append(_mask_row(f, 1, _lane_mask(0, 2 * f)))
        perturbed_gt.append(_mask_row(f, 2, _lane_mask(1, 30 - 2 * f)))
        perturbed_gt.append(_mask_row(f, 3, _lane_mask(2, 5 + f)))
        if f not in (3, 4):
            perturbed_pred.append(_mask_row(f, 101, _lane_mask(0, 2 * f + 1)))  # jittered
        if f <= 4:
            perturbed_pred.append(_mask_row(f, 102, _lane_mask(1, 30 - 2 * f)))
        else:
            perturbed_pred.append(_mask_row(f, 103, _lane_mask(1, 30 - 2 * f)))
        perturbed_pred.append(_mask_row(f, 104, _lane_mask(4, 3 * f)))

    # An ignore-region mask (class 10) fills lane 5; an unmatched prediction inside it
    # must be dropped identically on both sides before J&F runs.
    ignore_gt: list[MaskTrack] = []
    ignore_pred: list[MaskTrack] = []
    for f in range(1, 5):
        ignore_gt.append(_mask_row(f, 1, _lane_mask(0, 3 * f)))
        ignore_gt.append(
            _mask_row(f, 10, _lane_mask(5, 20, width=20), class_id=MOTS20_IGNORE_CLASS)
        )
        ignore_pred.append(_mask_row(f, 201, _lane_mask(0, 3 * f)))
        ignore_pred.append(_mask_row(f, 202, _lane_mask(5, 24, width=10)))  # inside ignore

    pred_only = [_mask_row(f, 301, _lane_mask(1, 4 * f)) for f in range(1, 6)]
    gt_only = [_mask_row(f, 1, _lane_mask(0, 4 * f)) for f in range(1, 6)]

    # Different track counts per sequence prove num_gt_tracks-weighted combining.
    seq_a_gt = [_mask_row(f, i, _lane_mask(i - 1, 2 * f)) for f in (1, 2, 3) for i in (1, 2)]
    seq_a_pred = [
        _mask_row(f, 100 + i, _lane_mask(i - 1, 2 * f)) for f in (1, 2, 3) for i in (1, 2)
    ]
    seq_b_gt = [_mask_row(f, 7, _lane_mask(0, 5 * f)) for f in (1, 2, 3, 4)]
    seq_b_pred = [_mask_row(f, 900, _lane_mask(0, 5 * f + 2)) for f in (1, 2)]  # partial, jittered

    # J&F casts its decay-bin edges to uint8 (upstream quirk), so they wrap past 255
    # frames. At 350 frames the edges (0-based timesteps) [0, 87, 175, 262, 349] wrap to
    # [0, 87, 175, 6, 93]: the last bin covers timesteps 6-93, not 262-349. (At 257-341
    # frames the last bin is empty and the decays are NaN; 350 keeps them finite.) The
    # prediction is exact for the first 50 frames and then shifted by 1-3 columns, so the
    # first bin, the wrapped last bin and the unwrapped last bin all have different means.
    long_frames = 350
    long_gt = [_mask_row(f, 1, _lane_mask(0, 20)) for f in range(1, long_frames + 1)]
    long_pred = [
        _mask_row(f, 401, _lane_mask(0, 20 if f <= 50 else 21 + f % 3))
        for f in range(1, long_frames + 1)
    ]

    return (
        Scenario(
            "jf_perturbed_predictions", (("SEQ-JF-01", frames, perturbed_gt, perturbed_pred),)
        ),
        Scenario("jf_ignore_region", (("SEQ-JF-02", 4, ignore_gt, ignore_pred),)),
        Scenario(
            "jf_empty_gt_and_empty_preds",
            (("SEQ-EMPTY-GT", 5, [], pred_only), ("SEQ-EMPTY-PRED", 5, gt_only, [])),
        ),
        Scenario(
            "jf_multi_sequence_combine",
            (("SEQ-COMB-A", 3, seq_a_gt, seq_a_pred), ("SEQ-COMB-B", 4, seq_b_gt, seq_b_pred)),
        ),
        Scenario("jf_decay_bins_wrap", (("SEQ-JF-LONG", long_frames, long_gt, long_pred),)),
    )


def write_mots_scenario(tmp_dir: Path, scenario: Scenario) -> dict[str, int]:
    """Write a MOTS scenario in the SKIP_SPLIT_FOL layout; returns ``{seq: length}``."""
    seq_lengths = {}
    for name, num_timesteps, gt_rows, pred_rows in scenario.sequences:
        write_mots(tmp_dir / "gt" / name / "gt" / "gt.txt", gt_rows)
        write_mots(predictions_dir(tmp_dir) / f"{name}.txt", pred_rows)
        seq_lengths[name] = num_timesteps
    return seq_lengths


def build_mots_dataset(tmp_dir: Path, scenario: Scenario) -> MOTDataset:
    """Build the moteval `MOTDataset` from the files `write_mots_scenario` wrote."""
    gt_sequences = []
    for name, num_timesteps, _, _ in scenario.sequences:
        rows = read_mots(tmp_dir / "gt" / name / "gt" / "gt.txt")
        gt_sequences.append(
            MaskGtSequence(
                name=name,
                num_timesteps=num_timesteps,
                tracks=tuple(t for t in rows if t.class_id != MOTS20_IGNORE_CLASS),
                ignore_regions=tuple(t for t in rows if t.class_id == MOTS20_IGNORE_CLASS),
            )
        )
    return MOTDataset(
        name="parity-mots", split="train", sequences=tuple(gt_sequences), protocol=MOTS20_PROTOCOL
    )


# jf_perturbed_predictions split by track id. Class 1: GT 1 with its jittered,
# gappy prediction 101 and the drifting false positive 104. Class 2: GT 2 (covered by
# 102, then 103) and the missed GT 3.
MOTS_MULTI_CLASS_TRACKS = {1: 1, 101: 1, 104: 1, 2: 2, 3: 2, 102: 2, 103: 2}


def mots_track_classes(_sequence: str, row: MaskTrack) -> int:
    return MOTS_MULTI_CLASS_TRACKS[row.track_id]


def write_multi_class_mots_scenario(tmp_dir: Path, scenario: Scenario, class_of: ClassOf) -> None:
    """Write a MOTS scenario with each row relabelled to its class from ``class_of``.

    The scenario must hold no ignore-region rows: relabelling would turn them into tracks.
    """
    for name, _, gt_rows, pred_rows in scenario.sequences:
        if any(t.class_id == MOTS20_IGNORE_CLASS for t in gt_rows):
            raise ValueError(f"{scenario.name}/{name}: ignore rows cannot be relabelled")
        write_mots(
            tmp_dir / "gt" / name / "gt" / "gt.txt",
            [replace(t, class_id=class_of(name, t)) for t in gt_rows],
        )
        write_mots(
            predictions_dir(tmp_dir) / f"{name}.txt",
            [replace(t, class_id=class_of(name, t)) for t in pred_rows],
        )


def build_multi_class_mots_dataset(tmp_dir: Path, scenario: Scenario) -> MOTDataset:
    """Build the two-class mask `MOTDataset` from `write_multi_class_mots_scenario` output."""
    gt_sequences = tuple(
        MaskGtSequence(
            name=name,
            num_timesteps=num_timesteps,
            tracks=tuple(read_mots(tmp_dir / "gt" / name / "gt" / "gt.txt")),
        )
        for name, num_timesteps, _, _ in scenario.sequences
    )
    return MOTDataset(
        name="parity-mots-multi-class",
        split="train",
        sequences=gt_sequences,
        protocol=MULTI_CLASS_MOTS_PROTOCOL,
    )


# ---------------------------------------------------------------------------
# TrackMAP scenarios (pure Python specs)
# ---------------------------------------------------------------------------

GtTracks = dict[int, dict[int, Sequence[float]]]
PredTracks = dict[int, dict[int, tuple[Sequence[float], float]]]


def box_rows_to_trackmap_spec(gt_rows: list, pred_rows: list) -> tuple[GtTracks, PredTracks]:
    """Convert MOT-txt value lists to the TrackMAP spec form (0-based frame keys).

    Frames are 1-indexed in the rows (`BOX_PROTOCOL`); values become floats, as a
    file reader produces them.
    """
    gt_tracks: GtTracks = {}
    for row in gt_rows:
        gt_tracks.setdefault(row[1], {})[row[0] - 1] = [float(v) for v in row[2:6]]
    pred_tracks: PredTracks = {}
    for row in pred_rows:
        pred_tracks.setdefault(row[1], {})[row[0] - 1] = (
            [float(v) for v in row[2:6]],
            float(row[6]),
        )
    return gt_tracks, pred_tracks


# name -> {seq_name: (num_timesteps, gt_tracks, pred_tracks)}. Ids here are moteval's
# view; the regen script offsets all ids by +1 on the oracle side so upstream's
# ``gt_m[thr, gt] > 0`` matched-id check (which misbehaves for a legitimate id of 0,
# fixed in moteval's TrackMAP) never fires.
TRACKMAP_SCENARIOS: dict[str, dict[str, tuple[int, GtTracks, PredTracks]]] = {
    "perfect_match": {
        "seq": (
            3,
            {1: {t: [10 + t, 10, 20, 20] for t in range(3)}},
            {101: {t: ([10 + t, 10, 20, 20], 0.9) for t in range(3)}},
        )
    },
    # seq1: gt 1 perfectly matched, gt 2 missed entirely, gt 3 (long, large area)
    # perfectly matched; a short false-positive pred track with no gt counterpart.
    # seq2: a partial-IoU match (box shifts out of overlap on the second frame).
    # Exercises all three area and time subsets simultaneously.
    "misses_false_positives_partial_iou_multi_sequence": {
        "seq1": (
            15,
            {
                1: {t: [10 + t, 10, 20, 20] for t in range(5)},
                2: {t: [200, 200, 15, 15] for t in range(5)},
                3: {t: [400, 400, 300, 300] for t in range(15)},
            },
            {
                101: {t: ([10 + t, 10, 20, 20], 0.9) for t in range(5)},
                102: {t: ([600, 600, 10, 10], 0.8) for t in range(3)},
                103: {t: ([400, 400, 300, 300], 0.95) for t in range(15)},
            },
        ),
        "seq2": (
            2,
            {10: {0: [50, 50, 50, 50], 1: [52, 50, 50, 50]}},
            {201: {0: ([50, 50, 50, 50], 0.7), 1: ([90, 50, 50, 50], 0.7)}},
        ),
    },
    # area_s = [0, 32**2], area_m = [32**2, 96**2]: a track with area exactly 32*32
    # must land in BOTH subsets (upstream's ranges are inclusive on both ends).
    "area_boundary_straddle": {
        "seq": (
            2,
            {1: {t: [0, 0, 32, 32] for t in range(2)}},
            {101: {t: ([0, 0, 32, 32], 0.9) for t in range(2)}},
        )
    },
    # time_s = [0, 3], time_m = [3, 10]: a track of length exactly 3 lands in both.
    "time_boundary_straddle": {
        "seq": (
            3,
            {1: {t: [10, 10, 5, 5] for t in range(3)}},
            {101: {t: ([10, 10, 5, 5], 0.9) for t in range(3)}},
        )
    },
    "empty_gt": {"seq": (2, {}, {201: {0: ([1, 1, 1, 1], 0.5)}})},
    "empty_preds": {"seq": (2, {1: {0: [1, 1, 1, 1]}}, {})},
    "both_empty": {"seq": (2, {}, {})},
    # moteval's densified ids always start at 0 for both gt and pred; its boolean
    # matched-state fix must agree with the oracle run on offset (never-0) ids.
    "id_zero_hazard": {
        "seq": (
            3,
            {
                0: {t: [10, 10, 20, 20] for t in range(3)},
                1: {t: [10, 10, 20, 20] for t in range(3)},  # identical box: ties with id 0
            },
            {
                0: {t: ([10, 10, 20, 20], 0.99) for t in range(3)},
                1: {t: ([10, 10, 20, 20], 0.5) for t in range(3)},
            },
        )
    },
    # One gt track, two dt tracks both perfectly overlapping it (an IoU tie): the LOW
    # id carries the LOW score, so only a descending-score dt presort wins the match
    # with the high-score track (AP ~1.0 instead of 0.5).
    "dt_track_order_score_presort": {
        "seq": (
            3,
            {5: {t: [10, 10, 20, 20] for t in range(3)}},
            {
                1: {t: ([10, 10, 20, 20], 0.5) for t in range(3)},  # low score, low id
                9: {t: ([10, 10, 20, 20], 0.9) for t in range(3)},  # high score, high id
            },
        )
    },
    # Track IoU sums each frame's intersection and union in the iteration order of
    # upstream's `set(gt.keys()) | set(dt.keys())`. For this sparse pair that order is
    # [23, 5, 7, 9, 12], not ascending, so moteval's vectorized IoU must fall back to the
    # scalar `_track_iou`. The summation order changes the last bit of the IoU: upstream's
    # order gives 0.5499999999999998, a match at the 0.55 threshold (`iou < thr - eps` is
    # false); ascending order and the `set(gt) | set(dt)` order ([7, 5, 23, 9, 12]) give
    # 0.5499999999999997, a miss. The height of dt frame 12 was searched to sit on that
    # boundary. Frame keys stay ascending, as upstream's loader inserts them; for this
    # pair no insertion order changes the set order (`track_iou_insertion_order` does).
    "track_iou_frame_order": {
        "seq": (
            24,
            {1: {5: [10.0, 10.0, 20.3, 30.7], 9: [12.0, 10.0, 20.3, 30.7]}},
            {
                101: {
                    5: ([11.1, 10.0, 20.3, 30.7], 0.9),
                    7: ([40.0, 40.0, 1.0, 359.1], 0.9),
                    9: ([13.7, 11.0, 20.3, 30.7], 0.9),
                    12: ([70.0, 70.0, 1.0, 145.32909090909212], 0.9),
                    23: ([60.0, 60.0, 1.0, 220.7], 0.9),
                }
            },
        )
    },
    # Frame keys 1 and 17 collide in the union set's hash table, so the set order depends
    # on insertion order: ascending insertion (upstream's TAO loader sorts by frame) gives
    # [1, 20, 17], reversed insertion [17, 20, 1]. Frame 1 overlaps, frame 17 is GT only,
    # and the height of dt frame 20 was searched to sit on the 0.55 boundary: ascending
    # order gives 0.5499999999999997 (a miss), reversed order 0.5499999999999998 (a match).
    "track_iou_insertion_order": {
        "seq": (
            24,
            {1: {1: [10.0, 10.0, 20.3, 30.7], 17: [12.0, 10.0, 3.3, 7.1]}},
            {
                101: {
                    1: ([11.1, 10.0, 20.3, 30.7], 0.9),
                    20: ([60.0, 60.0, 1.0, 391.2990909090918], 0.9),
                }
            },
        )
    },
}


def build_trackmap_sequence_data(
    num_timesteps: int, gt_tracks: GtTracks, pred_tracks: PredTracks
) -> SequenceData:
    """Build moteval's `SequenceData` from a TrackMAP scenario spec."""
    gt_ids_sorted = sorted(gt_tracks)
    gt_id_map = {tid: i for i, tid in enumerate(gt_ids_sorted)}
    pred_ids_sorted = sorted(pred_tracks)
    pred_id_map = {tid: i for i, tid in enumerate(pred_ids_sorted)}

    gt_ids: list[list[int]] = [[] for _ in range(num_timesteps)]
    gt_boxes: list[list] = [[] for _ in range(num_timesteps)]
    pred_ids: list[list[int]] = [[] for _ in range(num_timesteps)]
    pred_boxes: list[list] = [[] for _ in range(num_timesteps)]
    pred_confs: list[list[float]] = [[] for _ in range(num_timesteps)]

    for tid, frames in gt_tracks.items():
        for t, box in frames.items():
            gt_ids[t].append(gt_id_map[tid])
            gt_boxes[t].append(box)
    for tid, frames in pred_tracks.items():
        for t, (box, conf) in frames.items():
            pred_ids[t].append(pred_id_map[tid])
            pred_boxes[t].append(box)
            pred_confs[t].append(conf)

    gt_ids_arr = tuple(np.array(f, dtype=np.int64) for f in gt_ids)
    gt_boxes_arr = tuple(np.array(f, dtype=np.float64).reshape(-1, 4) for f in gt_boxes)
    pred_ids_arr = tuple(np.array(f, dtype=np.int64) for f in pred_ids)
    pred_boxes_arr = tuple(np.array(f, dtype=np.float64).reshape(-1, 4) for f in pred_boxes)
    pred_confs_arr = tuple(np.array(f, dtype=np.float64) for f in pred_confs)
    similarity = tuple(box_iou(g, p) for g, p in zip(gt_boxes_arr, pred_boxes_arr, strict=True))

    return SequenceData(
        name="synthetic",
        num_timesteps=num_timesteps,
        num_gt_ids=len(gt_ids_sorted),
        num_pred_ids=len(pred_ids_sorted),
        num_gt_dets=sum(a.shape[0] for a in gt_boxes_arr),
        num_pred_dets=sum(a.shape[0] for a in pred_boxes_arr),
        gt_ids=gt_ids_arr,
        pred_ids=pred_ids_arr,
        pred_confidences=pred_confs_arr,
        geometry=BoxGeometry(gt=gt_boxes_arr, pred=pred_boxes_arr),
        similarity=similarity,
    )


# ---------------------------------------------------------------------------
# Real-data cases (seeded perturbed predictions against `default_data_root()`)
# ---------------------------------------------------------------------------


def _write_perturbed_box_predictions(dataset: MOTDataset, tmp_dir: Path) -> dict[str, int]:
    pred_dir = predictions_dir(tmp_dir)
    seq_lengths = {}
    for i, seq in enumerate(dataset.sequences):
        preds = perturb_box_tracks(
            seq.tracks,
            seq.num_timesteps,
            dataset.protocol.frame_convention,
            seed=REAL_DATA_SEED + i,
        )
        write_mot(pred_dir / f"{seq.name}.txt", preds)
        seq_lengths[seq.name] = seq.num_timesteps
    return seq_lengths


def prepare_dancetrack_val(tmp_dir: Path) -> tuple[MOTDataset, Path, dict[str, int]]:
    """Load DanceTrack val, write seeded perturbed predictions into ``tmp_dir``.

    Returns ``(dataset, gt_root, seq_lengths)``; predictions land in
    ``predictions_dir(tmp_dir)``.
    """
    root = default_data_root() / "dancetrack"
    dataset = load_dancetrack(root=root, split="val")
    return dataset, root / "val", _write_perturbed_box_predictions(dataset, tmp_dir)


def prepare_sportsmot_val(tmp_dir: Path) -> tuple[MOTDataset, Path, dict[str, int]]:
    """Same contract as `prepare_dancetrack_val`, for SportsMOT val."""
    root = default_data_root() / "sportsmot"
    dataset = load_sportsmot(root=root, split="val")
    return dataset, root / "val", _write_perturbed_box_predictions(dataset, tmp_dir)


def prepare_mots20_sequence(tmp_dir: Path) -> tuple[MOTDataset, Path, dict[str, int]]:
    """Load MOTS20 train restricted to its first sequence, with seeded mask predictions."""
    root = default_data_root() / "mots20"
    dataset = load_mots20(root=root, split="train")
    seq = dataset.sequences[0]
    dataset = type(dataset)(
        name=dataset.name, split=dataset.split, sequences=(seq,), protocol=dataset.protocol
    )
    preds = perturb_mask_tracks(
        seq.tracks, seq.num_timesteps, dataset.protocol.frame_convention, seed=REAL_DATA_SEED
    )
    write_mots(predictions_dir(tmp_dir) / f"{seq.name}.txt", preds)
    return dataset, root / "train", {seq.name: seq.num_timesteps}
