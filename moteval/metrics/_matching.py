"""The machine-epsilon guard shared by every metric.

TrackEval compares similarities against thresholds with an ``np.finfo('float').eps``
guard (``>= alpha - eps`` in HOTA, ``< threshold - eps`` in CLEAR, and so on).
Every metric imports this one constant, so those comparisons match upstream
bit-for-bit. Metrics call scipy's ``linear_sum_assignment`` directly.
"""

import numpy as np

EPS = np.finfo(float).eps
