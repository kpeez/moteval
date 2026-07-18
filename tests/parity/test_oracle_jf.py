"""Parity: moteval J&F (and mask HOTA/CLEAR/Identity/Count) vs the TrackEval oracle.

Both sides read the identical on-disk MOTS-format fixtures (whitespace rows
``frame id class img_h img_w rle``). Predictions carry track ids disjoint from
GT ids. Masks live in disjoint row "lanes" so no frame ever contains
overlapping masks (the oracle rejects those). Every reported field of every
metric must be exactly equal — J&F included, which exercises the zero-mask
padding (pred tracks missing in some frames, whole missing pred tracks) and
the decay-bin quirks.

The oracle's JAndF imports cv2 and scikit-image at eval time; they are dev-only
dependencies. moteval's J&F itself needs neither.
"""

from pathlib import Path

import numpy as np
import pytest

from moteval import CLEAR, HOTA, Count, Identity, MOTDataset, evaluate
from moteval.benchmarks.mots20 import MOTS20_IGNORE_CLASS, MOTS20_PROTOCOL
from moteval.data.model import MaskGtSequence
from moteval.data.similarity import encode_mask
from moteval.formats.mots_txt import MaskTrack, read_mots, write_mots
from moteval.metrics.jf import JAndF
from tests.oracle.runner import run_mots_challenge

pytest.importorskip("cv2", reason="oracle JAndF needs cv2 (dev dependency)")
pytest.importorskip("skimage", reason="oracle JAndF needs scikit-image (dev dependency)")

H, W = 48, 64
JF_FIELDS = ("J-Mean", "J-Recall", "J-Decay", "F-Mean", "F-Recall", "F-Decay", "J&F")


def _lane_mask(lane: int, col: int, width: int = 10) -> np.ndarray:
    """A 6-row-tall block in row lane ``lane`` starting at column ``col``.

    Lanes are 8 rows apart so masks in different lanes never overlap.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    r0 = lane * 8
    c0 = max(0, min(col, W - width))
    mask[r0 : r0 + 6, c0 : c0 + width] = 1
    return mask


def _row(frame: int, track_id: int, mask: np.ndarray, class_id: int = 2) -> MaskTrack:
    counts = encode_mask(mask)["counts"]
    assert isinstance(counts, bytes)
    return MaskTrack(
        frame=frame, track_id=track_id, class_id=class_id, img_h=H, img_w=W, rle=counts.decode()
    )


def _run_both(tmp_path: Path, sequences: list[tuple[str, int, list[MaskTrack], list[MaskTrack]]]):
    gt_sequences = []
    seq_lengths = {}
    for name, num_timesteps, gt_rows, pred_rows in sequences:
        gt_path = tmp_path / "gt" / name / "gt" / "gt.txt"
        pred_path = tmp_path / "trackers" / "oracle" / "data" / f"{name}.txt"
        write_mots(gt_path, gt_rows)
        write_mots(pred_path, pred_rows)
        rows = read_mots(gt_path)
        gt_sequences.append(
            MaskGtSequence(
                name=name,
                num_timesteps=num_timesteps,
                tracks=tuple(t for t in rows if t.class_id != MOTS20_IGNORE_CLASS),
                ignore_regions=tuple(t for t in rows if t.class_id == MOTS20_IGNORE_CLASS),
            )
        )
        seq_lengths[name] = num_timesteps

    dataset = MOTDataset(
        name="parity-mots",
        split="train",
        sequences=tuple(gt_sequences),
        protocol=MOTS20_PROTOCOL,
    )
    ours = evaluate(
        dataset,
        tmp_path / "trackers" / "oracle" / "data",
        [HOTA(), CLEAR(), Identity(), Count(), JAndF()],
    )
    oracle = run_mots_challenge(tmp_path / "gt", tmp_path / "trackers", seq_lengths)
    return ours.combined, oracle


def _assert_all_equal(ours, oracle) -> None:
    for metric in ("HOTA", "CLEAR", "Identity", "Count"):
        for field, oracle_value in oracle[metric].items():
            if field not in ours[metric]:
                continue  # oracle-only derived summary fields, none expected
            ours_value = ours[metric][field]
            assert np.array_equal(np.asarray(ours_value), np.asarray(oracle_value)), (
                f"{metric}.{field}: {ours_value} != {oracle_value}"
            )
    for field in (*JF_FIELDS, "num_gt_tracks"):
        ours_value = ours["JAndF"][field]
        oracle_value = oracle["JAndF"][field]
        assert np.array_equal(
            np.asarray(ours_value, dtype=float),
            np.asarray(oracle_value, dtype=float),
            equal_nan=True,
        ), f"JAndF.{field}: {ours_value} != {oracle_value}"


def test_jf_parity_perturbed_predictions(tmp_path):
    # Three GT tracks over 8 frames; predictions are independently numbered and
    # perturbed: track 101 covers GT 1 but skips frames 3-4 (padding path),
    # GT 2 is covered by 102 for frames 1-4 then 103 (ID switch), GT 3 is
    # entirely missed (tracker padding path), and 104 is a pure false positive
    # whose masks drift.
    frames = 8
    gt_rows, pred_rows = [], []
    for f in range(1, frames + 1):
        gt_rows.append(_row(f, 1, _lane_mask(0, 2 * f)))
        gt_rows.append(_row(f, 2, _lane_mask(1, 30 - 2 * f)))
        gt_rows.append(_row(f, 3, _lane_mask(2, 5 + f)))
        if f not in (3, 4):
            pred_rows.append(_row(f, 101, _lane_mask(0, 2 * f + 1)))  # jittered
        if f <= 4:
            pred_rows.append(_row(f, 102, _lane_mask(1, 30 - 2 * f)))
        else:
            pred_rows.append(_row(f, 103, _lane_mask(1, 30 - 2 * f)))
        pred_rows.append(_row(f, 104, _lane_mask(4, 3 * f)))
    ours, oracle = _run_both(tmp_path, [("SEQ-JF-01", frames, gt_rows, pred_rows)])
    _assert_all_equal(ours, oracle)


def test_jf_parity_with_ignore_region(tmp_path):
    # An ignore-region mask (class 10) fills lane 5; an unmatched prediction
    # inside it must be dropped identically on both sides before J&F runs.
    frames = 4
    gt_rows, pred_rows = [], []
    for f in range(1, frames + 1):
        gt_rows.append(_row(f, 1, _lane_mask(0, 3 * f)))
        gt_rows.append(_row(f, 10, _lane_mask(5, 20, width=20), class_id=MOTS20_IGNORE_CLASS))
        pred_rows.append(_row(f, 201, _lane_mask(0, 3 * f)))
        pred_rows.append(_row(f, 202, _lane_mask(5, 24, width=10)))  # inside ignore
    ours, oracle = _run_both(tmp_path, [("SEQ-JF-02", frames, gt_rows, pred_rows)])
    _assert_all_equal(ours, oracle)
    assert ours["Count"]["Dets"] == 4.0  # 202 dropped every frame


def test_jf_parity_empty_gt_and_empty_preds(tmp_path):
    frames = 5
    pred_only = [_row(f, 301, _lane_mask(1, 4 * f)) for f in range(1, frames + 1)]
    gt_only = [_row(f, 1, _lane_mask(0, 4 * f)) for f in range(1, frames + 1)]
    ours, oracle = _run_both(
        tmp_path,
        [
            ("SEQ-EMPTY-GT", frames, [], pred_only),
            ("SEQ-EMPTY-PRED", frames, gt_only, []),
        ],
    )
    _assert_all_equal(ours, oracle)


def test_jf_parity_multi_sequence_combine(tmp_path):
    # Different track counts per sequence prove num_gt_tracks-weighted combining.
    seq_a_gt = [_row(f, i, _lane_mask(i - 1, 2 * f)) for f in (1, 2, 3) for i in (1, 2)]
    seq_a_pred = [_row(f, 100 + i, _lane_mask(i - 1, 2 * f)) for f in (1, 2, 3) for i in (1, 2)]
    seq_b_gt = [_row(f, 7, _lane_mask(0, 5 * f)) for f in (1, 2, 3, 4)]
    seq_b_pred = [_row(f, 900, _lane_mask(0, 5 * f + 2)) for f in (1, 2)]  # partial, jittered
    ours, oracle = _run_both(
        tmp_path,
        [("SEQ-COMB-A", 3, seq_a_gt, seq_a_pred), ("SEQ-COMB-B", 4, seq_b_gt, seq_b_pred)],
    )
    _assert_all_equal(ours, oracle)
