"""Shared matching primitives reused by HOTA, CLEAR, and Identity.

Mirrors TrackEval's use of scipy's Hungarian solver with an eps-guarded
threshold comparison (``np.finfo('float').eps``), so alpha/threshold
comparisons match upstream bit-for-bit.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

EPS = np.finfo(float).eps


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the assignment problem; thin wrapper kept for a single import site."""
    return _linear_sum_assignment(cost_matrix)
