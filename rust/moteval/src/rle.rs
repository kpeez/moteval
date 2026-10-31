//! pycocotools-compatible run-length encoded masks.
//!
//! The rle-masks-jf package ports pycocotools 2.0.11 `common/maskApi.c` and the parts of
//! `_mask.pyx` moteval calls: `encode` (Fortran order, `(h, w, n)` input only), `decode`,
//! `area`, `to_bbox`, `merge` (union and intersect), `iou` (with `iscrowd`), and the
//! compressed counts string. The representation inside `RleMask` is that package's to
//! choose; the constructor and accessors below are the shared contract.

use std::fmt;

/// One mask in pycocotools RLE form: `{"size": [height, width], "counts": bytes}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RleMask {
    height: usize,
    width: usize,
    counts: Vec<u8>,
}

impl RleMask {
    /// A mask from its compressed counts string (the `counts` bytes pycocotools stores).
    pub fn from_compressed(height: usize, width: usize, counts: &[u8]) -> Result<Self, RleError> {
        Ok(Self {
            height,
            width,
            counts: counts.to_vec(),
        })
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// The compressed counts string, as pycocotools writes it.
    pub fn to_compressed(&self) -> Vec<u8> {
        self.counts.clone()
    }
}

/// Errors from RLE decoding and mask operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RleError {
    /// A counts string that does not decode to `height * width` pixels.
    InvalidCounts,
    /// Masks of different sizes passed to one `merge` or `iou` call.
    SizeMismatch,
}

impl fmt::Display for RleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCounts => f.write_str("invalid RLE counts string"),
            Self::SizeMismatch => f.write_str("RLE masks differ in size"),
        }
    }
}

impl std::error::Error for RleError {}

/// `pycocotools.mask.encode` of one Fortran-order `height x width` binary mask.
pub fn encode(_mask_fortran: &[u8], _height: usize, _width: usize) -> RleMask {
    todo!("rle-masks-jf: port rleEncode")
}

/// `pycocotools.mask.decode`: the binary mask in Fortran order.
pub fn decode(_rle: &RleMask) -> Vec<u8> {
    todo!("rle-masks-jf: port rleDecode")
}

/// `pycocotools.mask.area`: the number of foreground pixels.
pub fn area(_rle: &RleMask) -> u64 {
    todo!("rle-masks-jf: port rleArea")
}

/// `pycocotools.mask.toBbox`: `[x, y, w, h]`.
pub fn to_bbox(_rle: &RleMask) -> [f64; 4] {
    todo!("rle-masks-jf: port rleToBbox")
}

/// `pycocotools.mask.merge(rles, intersect)`.
pub fn merge(_rles: &[RleMask], _intersect: bool) -> Result<RleMask, RleError> {
    todo!("rle-masks-jf: port rleMerge")
}

/// `pycocotools.mask.iou(dt, gt, iscrowd)`: a `dt.len() x gt.len()` matrix in C order.
pub fn iou(_dt: &[RleMask], _gt: &[RleMask], _iscrowd: &[bool]) -> Result<Vec<f64>, RleError> {
    todo!("rle-masks-jf: port rleIou")
}
