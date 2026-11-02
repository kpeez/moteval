from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from moteval.benchmarks import default_data_root, load_dataset
from moteval.data.model import (
    BoxGeometry,
    FrameConvention,
    GtSequence,
    MaskGtSequence,
    MOTDataset,
    SequenceData,
)
from moteval.data.protocol import Protocol
from moteval.formats import MaskTrack, read_mot, read_mots, write_mot, write_mots
from tests.perturb import perturb_box_tracks, perturb_mask_tracks
from tests.rle import encode_mask

MOTS20_IGNORE_CLASS = 10
MOTS20_PROTOCOL = Protocol(
    name="mots20",
    frame_convention=FrameConvention(name="1-indexed", first_frame=1),
    eval_classes=(2,),
    drop_zero_conf_gt=False,
    matching_fill=-10000.0,
)


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)
    a = boxes_a.astype(np.float64)
    b = boxes_b.astype(np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    # TrackEval takes areas from the corners: (x0 + w) - x0 can differ from w by an ulp.
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    eps = np.finfo("float").eps
    inter[area_a <= eps, :] = 0.0
    inter[:, area_b <= eps] = 0.0
    inter[union <= eps] = 0.0
    union[union <= eps] = 1.0
    return inter / union


BOX_PROTOCOL = Protocol(
    name="parity",
    frame_convention=FrameConvention(name="1-indexed", first_frame=1),
    eval_classes=(1,),
)
BOX_METRICS = ("HOTA", "CLEAR", "Identity", "Count")
COMBINE_CLASSES_SCENARIO = "combine_classes"
MOTS_MULTI_CLASS_SCENARIO = "jf_perturbed_predictions"

REAL_DATA_SEED = 20260718


@dataclass(frozen=True)
class Scenario:
    name: str
    sequences: tuple[tuple[str, int, list, list], ...]


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
    Scenario("misses", (("MISS01", 3, _MISS_GT, _MISS_PRED),)),
    Scenario("false_positives", (("FP01", 2, _FP_GT, _FP_PRED),)),
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
    # Every pair has IoU 1.0, so scipy's linear_sum_assignment tie-breaking picks the matches.
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
    Scenario(
        "single_frame_track",
        (("SINGLE01", 1, [[1, 1, 10, 10, 20, 40, 1, 1, 1]], [[1, 701, 10, 10, 20, 40, 1]]),),
    ),
    # GT 2 and pred 950 keep frame 3 non-empty: TrackEval's CLEAR skips a frame with no GT
    # or no predictions without resetting its Frag state, so the gap would count no Frag.
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
    # TrackEval's Identity (G+T)x(G+T) block cost matrix with G != T, in both directions.
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
    seq_lengths = {}
    for name, num_timesteps, gt_rows, pred_rows in scenario.sequences:
        _write_box_rows(tmp_dir / "gt" / name / "gt" / "gt.txt", gt_rows)
        _write_box_rows(predictions_dir(tmp_dir) / f"{name}.txt", pred_rows)
        seq_lengths[name] = num_timesteps
    return seq_lengths


def build_box_dataset(tmp_dir: Path, scenario: Scenario) -> MOTDataset:
    gt_sequences = tuple(
        GtSequence(
            name=name,
            num_timesteps=num_timesteps,
            tracks=tuple(read_mot(tmp_dir / "gt" / name / "gt" / "gt.txt")),
        )
        for name, num_timesteps, _, _ in scenario.sequences
    )
    return MOTDataset(name="synthetic", split="val", sequences=gt_sequences, protocol=BOX_PROTOCOL)


MULTI_CLASS_BOX_PROTOCOL = replace(BOX_PROTOCOL, name="parity-multi-class", eval_classes=(1, 2))
MULTI_CLASS_MOTS_PROTOCOL = replace(
    MOTS20_PROTOCOL, name="parity-mots-multi-class", eval_classes=(1, 2)
)
ClassOf = Callable[[str, Any], int]


def sequence_classes(scenario: Scenario) -> ClassOf:
    index = {name: i + 1 for i, (name, _, _, _) in enumerate(scenario.sequences)}
    return lambda sequence, _row: index[sequence]


def class_views(
    scenario: Scenario, class_of: ClassOf, classes: tuple[int, ...]
) -> tuple[Scenario, ...]:
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


MASK_H, MASK_W = 48, 64


def _lane_mask(lane: int, col: int, width: int = 10) -> np.ndarray:
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
    frames = 8
    perturbed_gt: list[MaskTrack] = []
    perturbed_pred: list[MaskTrack] = []
    for f in range(1, frames + 1):
        perturbed_gt.append(_mask_row(f, 1, _lane_mask(0, 2 * f)))
        perturbed_gt.append(_mask_row(f, 2, _lane_mask(1, 30 - 2 * f)))
        perturbed_gt.append(_mask_row(f, 3, _lane_mask(2, 5 + f)))
        if f not in (3, 4):
            perturbed_pred.append(_mask_row(f, 101, _lane_mask(0, 2 * f + 1)))
        if f <= 4:
            perturbed_pred.append(_mask_row(f, 102, _lane_mask(1, 30 - 2 * f)))
        else:
            perturbed_pred.append(_mask_row(f, 103, _lane_mask(1, 30 - 2 * f)))
        perturbed_pred.append(_mask_row(f, 104, _lane_mask(4, 3 * f)))

    ignore_gt: list[MaskTrack] = []
    ignore_pred: list[MaskTrack] = []
    for f in range(1, 5):
        ignore_gt.append(_mask_row(f, 1, _lane_mask(0, 3 * f)))
        ignore_gt.append(
            _mask_row(f, 10, _lane_mask(5, 20, width=20), class_id=MOTS20_IGNORE_CLASS)
        )
        ignore_pred.append(_mask_row(f, 201, _lane_mask(0, 3 * f)))
        ignore_pred.append(_mask_row(f, 202, _lane_mask(5, 24, width=10)))

    pred_only = [_mask_row(f, 301, _lane_mask(1, 4 * f)) for f in range(1, 6)]
    gt_only = [_mask_row(f, 1, _lane_mask(0, 4 * f)) for f in range(1, 6)]

    seq_a_gt = [_mask_row(f, i, _lane_mask(i - 1, 2 * f)) for f in (1, 2, 3) for i in (1, 2)]
    seq_a_pred = [
        _mask_row(f, 100 + i, _lane_mask(i - 1, 2 * f)) for f in (1, 2, 3) for i in (1, 2)
    ]
    seq_b_gt = [_mask_row(f, 7, _lane_mask(0, 5 * f)) for f in (1, 2, 3, 4)]
    seq_b_pred = [_mask_row(f, 900, _lane_mask(0, 5 * f + 2)) for f in (1, 2)]

    # TrackEval's `_combine_weighted_av` adds these eight unequal np.float64 products with
    # builtin `sum`, left to right; CPython compensates only exact floats.
    many_sequences = []
    for i in range(8):
        seq_frames = 4 + i
        tracks = 1 + i % 3
        many_gt = [
            _mask_row(f, lane + 1, _lane_mask(lane, 3 * f + 2 * lane))
            for f in range(1, seq_frames + 1)
            for lane in range(tracks)
        ]
        many_pred = [
            _mask_row(f, 100 + lane, _lane_mask(lane, 3 * f + 2 * lane + (f * (i + 1) + lane) % 5))
            for f in range(1, seq_frames + 1)
            for lane in range(tracks)
            if (f + lane + i) % 4
        ]
        many_sequences.append((f"SEQ-MANY-{i}", seq_frames, many_gt, many_pred))

    # TrackEval's J&F casts its decay-bin edges to uint8, so at 350 frames the edges
    # [0, 87, 175, 262, 349] wrap to [0, 87, 175, 6, 93].
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
        Scenario("jf_many_sequence_combine", tuple(many_sequences)),
        Scenario("jf_decay_bins_wrap", (("SEQ-JF-LONG", long_frames, long_gt, long_pred),)),
    )


def write_mots_scenario(tmp_dir: Path, scenario: Scenario) -> dict[str, int]:
    seq_lengths = {}
    for name, num_timesteps, gt_rows, pred_rows in scenario.sequences:
        write_mots(tmp_dir / "gt" / name / "gt" / "gt.txt", gt_rows)
        write_mots(predictions_dir(tmp_dir) / f"{name}.txt", pred_rows)
        seq_lengths[name] = num_timesteps
    return seq_lengths


def build_mots_dataset(tmp_dir: Path, scenario: Scenario) -> MOTDataset:
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


MOTS_MULTI_CLASS_TRACKS = {1: 1, 101: 1, 104: 1, 2: 2, 3: 2, 102: 2, 103: 2}


def mots_track_classes(_sequence: str, row: MaskTrack) -> int:
    return MOTS_MULTI_CLASS_TRACKS[row.track_id]


def write_multi_class_mots_scenario(tmp_dir: Path, scenario: Scenario, class_of: ClassOf) -> None:
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


GtTracks = dict[int, dict[int, Sequence[float]]]
PredTracks = dict[int, dict[int, tuple[Sequence[float], float]]]


def box_rows_to_trackmap_spec(gt_rows: list, pred_rows: list) -> tuple[GtTracks, PredTracks]:
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


TRACKMAP_SCENARIOS: dict[str, dict[str, tuple[int, GtTracks, PredTracks]]] = {
    "perfect_match": {
        "seq": (
            3,
            {1: {t: [10 + t, 10, 20, 20] for t in range(3)}},
            {101: {t: ([10 + t, 10, 20, 20], 0.9) for t in range(3)}},
        )
    },
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
    # TrackEval's area ranges include both ends: area 32**2 is in area_s and area_m.
    "area_boundary_straddle": {
        "seq": (
            2,
            {1: {t: [0, 0, 32, 32] for t in range(2)}},
            {101: {t: ([0, 0, 32, 32], 0.9) for t in range(2)}},
        )
    },
    # TrackEval's time ranges include both ends: length 3 is in time_s and time_m.
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
    # TrackEval's `gt_m[thr, gt] > 0` check reads a match to dt id 0 as no match; the regen
    # script scores the oracle on ids offset by +1.
    "id_zero_hazard": {
        "seq": (
            3,
            {
                0: {t: [10, 10, 20, 20] for t in range(3)},
                1: {t: [10, 10, 20, 20] for t in range(3)},
            },
            {
                0: {t: ([10, 10, 20, 20], 0.99) for t in range(3)},
                1: {t: ([10, 10, 20, 20], 0.5) for t in range(3)},
            },
        )
    },
    # An IoU tie that TrackEval's TAO loader breaks by sorting dt tracks by descending score.
    "dt_track_order_score_presort": {
        "seq": (
            3,
            {5: {t: [10, 10, 20, 20] for t in range(3)}},
            {
                1: {t: ([10, 10, 20, 20], 0.5) for t in range(3)},
                9: {t: ([10, 10, 20, 20], 0.9) for t in range(3)},
            },
        )
    },
    # The same twelve scores, reordered: np.mean's pairwise sum gives track 1
    # 0.7687499999999999 and track 9 0.7687500000000002, so track 9 sorts first.
    "dt_track_score_pairwise_mean": {
        "seq": (
            12,
            {5: {t: [10, 10, 20, 20] for t in range(12)}},
            {
                1: {
                    t: ([10, 10, 20, 20], c)
                    for t, c in enumerate(
                        [0.73, 0.531, 0.821, 0.926, 0.796, 0.63, 0.92, 0.755, 0.755, 0.877]
                        + [0.574, 0.91]
                    )
                },
                9: {
                    t: ([14, 10, 20, 20], c)
                    for t, c in enumerate(
                        [0.574, 0.821, 0.531, 0.755, 0.796, 0.755, 0.877, 0.63, 0.73, 0.92]
                        + [0.91, 0.926]
                    )
                },
            },
        )
    },
    # TrackEval sums track IoU in CPython's `set(gt.keys()) | set(dt.keys())` order, here
    # [23, 5, 7, 9, 12]: 0.5499999999999998, a match at 0.55. Ascending order gives
    # 0.5499999999999997, a miss.
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
    # Keys 1 and 17 collide in CPython's set hash table, so insertion order sets the order.
    # TrackEval's TAO loader inserts frames ascending: [1, 20, 17] gives 0.5499999999999997,
    # a miss at 0.55. Reversed insertion, [17, 20, 1], gives 0.5499999999999998, a match.
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
    root = default_data_root() / "dancetrack"
    dataset = load_dataset("dancetrack", root=root, split="val")
    return dataset, root / "val", _write_perturbed_box_predictions(dataset, tmp_dir)


def prepare_sportsmot_val(tmp_dir: Path) -> tuple[MOTDataset, Path, dict[str, int]]:
    root = default_data_root() / "sportsmot"
    dataset = load_dataset("sportsmot", root=root, split="val")
    return dataset, root / "val", _write_perturbed_box_predictions(dataset, tmp_dir)


def prepare_mots20_sequence(tmp_dir: Path) -> tuple[MOTDataset, Path, dict[str, int]]:
    root = default_data_root() / "mots20"
    dataset = cast(MOTDataset[MaskGtSequence], load_dataset("mots20", root=root, split="train"))
    seq = dataset.sequences[0]
    dataset = type(dataset)(
        name=dataset.name, split=dataset.split, sequences=(seq,), protocol=dataset.protocol
    )
    preds = perturb_mask_tracks(
        seq.tracks, seq.num_timesteps, dataset.protocol.frame_convention, seed=REAL_DATA_SEED
    )
    write_mots(predictions_dir(tmp_dir) / f"{seq.name}.txt", preds)
    return dataset, root / "train", {seq.name: seq.num_timesteps}
