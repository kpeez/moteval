"""Mask geometry: RLE round-trips, mask IoU/IoA, MOTS-txt parsing, MOTS20 end-to-end."""

import numpy as np
import pytest

from moteval import CLEAR, HOTA, Count, Identity, evaluate
from moteval.benchmarks.mots20 import MOTS20_PROTOCOL, load_mots20
from moteval.data.model import BoxGeometry, MaskGeometry
from moteval.data.similarity import decode_mask, encode_mask, mask_ioa, mask_iou
from moteval.formats import MaskTrack, read_mots, write_mots


def _square_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask


def _counts(mask: np.ndarray) -> str:
    counts = encode_mask(mask)["counts"]
    assert isinstance(counts, bytes)
    return counts.decode()


# ---------------------------------------------------------------- RLE encoding


def test_rle_round_trip_from_c_order_input():
    # np.zeros is C-contiguous; encode_mask must convert, not reject.
    mask = _square_mask(12, 15, 2, 7, 3, 9)
    assert mask.flags["C_CONTIGUOUS"]
    rle = encode_mask(mask)
    assert rle["size"] == [12, 15]
    assert np.array_equal(decode_mask(rle), mask)


def test_rle_round_trip_from_fortran_order_input():
    mask = np.asfortranarray(_square_mask(9, 4, 1, 5, 0, 3))
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_encode_mask_rejects_non_2d_input():
    with pytest.raises(ValueError, match="single"):
        encode_mask(np.zeros((3, 4, 2), dtype=np.uint8))


# ------------------------------------------------------------------- mask IoU


def test_mask_iou_matches_hand_computed_overlap():
    # A: rows 0-4 full width (50 px). B: all rows, cols 0-4 (50 px).
    # Intersection 25 px, union 75 px -> IoU = 1/3.
    a = encode_mask(_square_mask(10, 10, 0, 5, 0, 10))
    b = encode_mask(_square_mask(10, 10, 0, 10, 0, 5))
    ious = mask_iou([a], [b])
    assert ious.shape == (1, 1)
    assert ious[0, 0] == pytest.approx(25 / 75)


def test_mask_iou_identical_and_disjoint():
    a = encode_mask(_square_mask(10, 10, 0, 5, 0, 5))
    far = encode_mask(_square_mask(10, 10, 6, 9, 6, 9))
    ious = mask_iou([a, far], [a])
    assert ious[0, 0] == 1.0
    assert ious[1, 0] == 0.0


def test_mask_iou_empty_inputs_keep_matrix_shape():
    a = encode_mask(_square_mask(10, 10, 0, 5, 0, 5))
    assert mask_iou([], [a]).shape == (0, 1)
    assert mask_iou([a], []).shape == (1, 0)
    assert mask_iou([], []).shape == (0, 0)


def test_mask_ioa_normalises_by_first_argument_area():
    # A (25 px) half-covered by B (50 px): IoA(A, B) = 25/25 = 1.0 when A is
    # inside B; here A is rows 0-4 cols 0-4, B is rows 0-9 cols 0-4 -> A fully
    # inside B -> IoA 1.0, while IoU would be 25/50.
    a = encode_mask(_square_mask(10, 10, 0, 5, 0, 5))
    b = encode_mask(_square_mask(10, 10, 0, 10, 0, 5))
    assert mask_ioa([a], [b])[0, 0] == pytest.approx(1.0)
    assert mask_iou([a], [b])[0, 0] == pytest.approx(0.5)


# ------------------------------------------------------------------- MOTS-txt


def test_read_mots_parses_and_round_trips(tmp_path):
    h, w = 12, 15
    rows = [
        MaskTrack(
            frame=1,
            track_id=5,
            class_id=2,
            img_h=h,
            img_w=w,
            rle=_counts(_square_mask(h, w, 0, 4, 0, 4)),
        ),
        MaskTrack(
            frame=2,
            track_id=5,
            class_id=2,
            img_h=h,
            img_w=w,
            rle=_counts(_square_mask(h, w, 4, 8, 4, 8)),
        ),
    ]
    path = tmp_path / "gt.txt"
    write_mots(path, rows)
    assert read_mots(path) == rows


def test_read_mots_rejects_malformed_rows(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 2 2 10\n")
    with pytest.raises(ValueError, match="malformed MOTS row"):
        read_mots(path)
    path.write_text("1 x 2 10 10 abc\n")
    with pytest.raises(ValueError, match="malformed MOTS row"):
        read_mots(path)


# ------------------------------------------------------------ MOTS20 benchmark


H, W = 20, 20
TRACK1 = [_square_mask(H, W, 2, 8, 2, 8), _square_mask(H, W, 3, 9, 2, 8)]
TRACK2 = [_square_mask(H, W, 12, 18, 12, 18), _square_mask(H, W, 12, 18, 11, 17)]


def _mask_row(frame: int, track_id: int, class_id: int, mask: np.ndarray) -> MaskTrack:
    return MaskTrack(
        frame=frame, track_id=track_id, class_id=class_id, img_h=H, img_w=W, rle=_counts(mask)
    )


def _write_mots20_layout(root, gt_rows):
    seq_dir = root / "train" / "MOTS20-02"
    write_mots(seq_dir / "gt" / "gt.txt", gt_rows)
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=MOTS20-02\nseqLength=2\n")


def _perfect_gt_rows():
    return [
        _mask_row(1, 1, 2, TRACK1[0]),
        _mask_row(2, 1, 2, TRACK1[1]),
        _mask_row(1, 2, 2, TRACK2[0]),
        _mask_row(2, 2, 2, TRACK2[1]),
    ]


def test_mots20_protocol_declares_mots_matching_semantics():
    assert MOTS20_PROTOCOL.eval_classes == (2,)
    assert MOTS20_PROTOCOL.matching_fill == -10000.0
    assert MOTS20_PROTOCOL.drop_zero_conf_gt is False
    assert MOTS20_PROTOCOL.frame_convention.first_frame == 1


def test_mots20_perfect_predictions_evaluate_end_to_end(tmp_path):
    _write_mots20_layout(tmp_path / "data", _perfect_gt_rows())
    dataset = load_mots20(root=tmp_path / "data", split="train")
    # Predictions identical to GT, but independently numbered (ids 7 and 9).
    preds = [
        _mask_row(1, 7, 2, TRACK1[0]),
        _mask_row(2, 7, 2, TRACK1[1]),
        _mask_row(1, 9, 2, TRACK2[0]),
        _mask_row(2, 9, 2, TRACK2[1]),
    ]
    pred_dir = tmp_path / "preds"
    write_mots(pred_dir / "MOTS20-02.txt", preds)

    result = evaluate(dataset, pred_dir, [HOTA(), CLEAR(), Identity(), Count()])

    scores = result.per_sequence["MOTS20-02"]
    assert scores["Count"] == {"Dets": 4.0, "GT_Dets": 4.0, "IDs": 2.0, "GT_IDs": 2.0}
    assert scores["CLEAR"]["MOTA"] == 1.0
    assert scores["Identity"]["IDF1"] == 1.0
    assert np.array_equal(scores["HOTA"]["HOTA"], np.ones(19))


def test_mots20_ignore_region_drops_unmatched_prediction(tmp_path):
    # GT: track 1 only. An ignore region (class 10) sits at rows 12-18.
    # Pred A matches GT track 1. Pred B is unmatched and fully inside the
    # ignore region -> dropped by preprocessing (IoA 1.0 > 0.5).
    gt_rows = [
        _mask_row(1, 1, 2, TRACK1[0]),
        _mask_row(2, 1, 2, TRACK1[1]),
        _mask_row(1, 100, 10, _square_mask(H, W, 12, 19, 10, 19)),
    ]
    _write_mots20_layout(tmp_path / "data", gt_rows)
    dataset = load_mots20(root=tmp_path / "data", split="train")
    seq = dataset.sequences[0]
    assert len(seq.ignore_regions) == 1
    assert len(seq.tracks) == 2

    preds = [
        _mask_row(1, 7, 2, TRACK1[0]),
        _mask_row(2, 7, 2, TRACK1[1]),
        _mask_row(1, 8, 2, TRACK2[0]),  # inside the frame-1 ignore region
        _mask_row(2, 8, 2, TRACK2[1]),  # frame 2 has no ignore region -> stays, FP
    ]
    pred_dir = tmp_path / "preds"
    write_mots(pred_dir / "MOTS20-02.txt", preds)

    result = evaluate(dataset, pred_dir, [Count(), CLEAR()])
    counts = result.per_sequence["MOTS20-02"]["Count"]
    assert counts == {"Dets": 3.0, "GT_Dets": 2.0, "IDs": 2.0, "GT_IDs": 1.0}
    assert result.per_sequence["MOTS20-02"]["CLEAR"]["CLR_FP"] == 1.0


def test_mots20_overlapping_gt_masks_raise(tmp_path):
    overlapping = [
        _mask_row(1, 1, 2, _square_mask(H, W, 2, 8, 2, 8)),
        _mask_row(1, 2, 2, _square_mask(H, W, 4, 10, 4, 10)),
        _mask_row(2, 1, 2, TRACK1[1]),
    ]
    _write_mots20_layout(tmp_path / "data", overlapping)
    dataset = load_mots20(root=tmp_path / "data", split="train")
    with pytest.raises(ValueError, match="overlapping GT masks"):
        evaluate(dataset, tmp_path / "preds", [Count()])


def test_mots20_out_of_range_prediction_frame_raises(tmp_path):
    _write_mots20_layout(tmp_path / "data", _perfect_gt_rows())
    dataset = load_mots20(root=tmp_path / "data", split="train")
    preds = [_mask_row(3, 7, 2, TRACK1[0])]  # seqLength is 2 -> frame 3 invalid
    pred_dir = tmp_path / "preds"
    write_mots(pred_dir / "MOTS20-02.txt", preds)
    with pytest.raises(ValueError, match="out of range"):
        evaluate(dataset, pred_dir, [Count()])


# ------------------------------------------------------------ geometry union


def test_geometry_accessors_fail_loudly_across_kinds(tmp_path):
    from moteval import load_dataset
    from moteval.data.convert import build_mask_sequence_data, build_sequence_data

    toy = load_dataset("toy")
    box_seq = toy.sequences[0]
    box_data = build_sequence_data(box_seq, (), toy.protocol, 1)
    assert isinstance(box_data.geometry, BoxGeometry)
    assert box_data.gt_boxes[0].shape[1] == 4
    with pytest.raises(TypeError, match="box geometry"):
        _ = box_data.gt_masks

    _write_mots20_layout(tmp_path / "data", _perfect_gt_rows())
    mots = load_mots20(root=tmp_path / "data", split="train")
    mask_data = build_mask_sequence_data(mots.sequences[0], (), mots.protocol, 2)
    assert isinstance(mask_data.geometry, MaskGeometry)
    assert mask_data.gt_masks[0].shape == (2,)
    with pytest.raises(TypeError, match="mask geometry"):
        _ = mask_data.gt_boxes
