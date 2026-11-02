import numpy as np
from pycocotools import mask as mask_utils

from moteval.data.model import RleMask


def encode_mask(mask: np.ndarray) -> RleMask:
    # pycocotools encodes only Fortran-contiguous uint8 arrays.
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
