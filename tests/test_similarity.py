import numpy as np

from moteval.data.similarity import box_iou


def test_box_iou_against_hand_computed():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10]], dtype=np.float64)
    ious = box_iou(a, b)
    # identical boxes -> 1.0; half-overlap -> inter 50 / union 150 = 1/3.
    np.testing.assert_allclose(ious, [[1.0, 1 / 3], [1.0, 1 / 3]])


def test_box_iou_disjoint_is_zero():
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[100, 100, 10, 10]], dtype=np.float64)
    assert box_iou(a, b)[0, 0] == 0.0


def test_box_iou_empty_sides():
    empty = np.zeros((0, 4), dtype=np.float64)
    full = np.array([[0, 0, 10, 10]], dtype=np.float64)
    assert box_iou(empty, full).shape == (0, 1)
    assert box_iou(full, empty).shape == (1, 0)
