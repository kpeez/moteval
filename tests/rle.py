"""Test-support RLE encoding for building mask fixtures from plain arrays."""

import numpy as np
from pycocotools import mask as mask_utils

from moteval.data.model import RleMask


def encode_mask(mask: np.ndarray) -> RleMask:
    """Encode one binary ``(h, w)`` mask as a compressed RLE dict.

    pycocotools requires Fortran-contiguous uint8 input, so C-order arrays
    (how fixtures naturally build masks) are converted first.
    """
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
