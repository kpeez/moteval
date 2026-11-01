//! Seeded binary masks and J&F sequences, draw for draw the same as
//! `scripts/testdata/maskgen.py`. A mask is `h * w` bytes, 0 or 1, in Fortran order.

use crate::harness::testgen::SplitMix64;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// FNV-1a over raw bytes (`maskgen.fnv1a64_bytes`).
pub fn fnv1a64_bytes(data: &[u8]) -> u64 {
    data.iter().fold(FNV_OFFSET, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
    })
}

/// FNV-1a over the little-endian 8-byte form of each value (`maskgen.fnv1a64_ints`).
pub fn fnv1a64_ints(values: &[u64]) -> u64 {
    let mut h = FNV_OFFSET;
    for v in values {
        for b in v.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
    }
    h
}

pub const FAMILIES: usize = 250;
pub const FAMILY_SIZE: usize = 4;

pub fn family_seed(index: usize) -> u64 {
    7_000_000 + index as u64
}

pub fn boundary_seed(index: usize) -> u64 {
    8_000_000 + index as u64
}

pub fn jf_sequence_seed(index: usize) -> u64 {
    9_000_000 + index as u64
}

fn below(rng: &mut SplitMix64, n: usize) -> usize {
    rng.below(n as u64) as usize
}

pub fn mask_size(rng: &mut SplitMix64) -> (usize, usize) {
    if below(rng, 8) == 0 {
        return (40 + below(rng, 160), 40 + below(rng, 160));
    }
    (1 + below(rng, 24), 1 + below(rng, 24))
}

fn fill(m: &mut [u8], h: usize, r0: usize, r1: usize, c0: usize, c1: usize, value: u8) {
    for c in c0..c1 {
        m[c * h + r0..c * h + r1].fill(value);
    }
}

fn rect(rng: &mut SplitMix64, h: usize, w: usize) -> (usize, usize, usize, usize) {
    let r0 = below(rng, h);
    let c0 = below(rng, w);
    let r1 = r0 + 1 + below(rng, h - r0);
    let c1 = c0 + 1 + below(rng, w - c0);
    (r0, r1, c0, c1)
}

pub fn random_mask(rng: &mut SplitMix64, h: usize, w: usize, allow_noise: bool) -> Vec<u8> {
    let mut m = vec![0u8; h * w];
    let mut kind = below(rng, 8);
    if kind == 4 && !allow_noise {
        kind = 3;
    }
    match kind {
        0 => {}
        1 => m.fill(1),
        2 => {
            let r = below(rng, h);
            let c = below(rng, w);
            m[c * h + r] = 1;
        }
        3 | 6 => {
            let mut value = 1;
            if kind == 6 {
                m.fill(1);
                value = 0;
            }
            for _ in 0..1 + below(rng, 4) {
                let (r0, r1, c0, c1) = rect(rng, h, w);
                fill(&mut m, h, r0, r1, c0, c1, value);
            }
        }
        4 => {
            let p = 1 + below(rng, 7);
            for px in m.iter_mut() {
                if below(rng, 8) < p {
                    *px = 1;
                }
            }
        }
        5 => {
            for c in 0..w {
                if below(rng, 3) != 0 {
                    let a = below(rng, h);
                    let b = a + below(rng, h - a + 1);
                    fill(&mut m, h, a, b, c, c + 1, 1);
                }
            }
        }
        _ => {
            for _ in 0..1 + below(rng, 3) {
                let (mut r0, mut r1, mut c0, mut c1) = rect(rng, h, w);
                match below(rng, 4) {
                    0 => r0 = 0,
                    1 => r1 = h,
                    2 => c0 = 0,
                    _ => c1 = w,
                }
                fill(&mut m, h, r0, r1, c0, c1, 1);
            }
        }
    }
    m
}

pub fn mask_area(m: &[u8]) -> usize {
    m.iter().map(|&v| usize::from(v)).sum()
}

fn nonempty_mask(rng: &mut SplitMix64, h: usize, w: usize) -> Vec<u8> {
    loop {
        let m = random_mask(rng, h, w, true);
        let area = mask_area(&m);
        if 0 < area && area < h * w {
            return m;
        }
    }
}

/// One mask with its size.
pub struct SizedMask {
    pub h: usize,
    pub w: usize,
    pub bytes: Vec<u8>,
}

/// `maskgen.boundary_case`: two masks with non-empty boundaries and a radius.
pub fn boundary_case(index: usize) -> (SizedMask, SizedMask, i64) {
    let mut rng = SplitMix64::new(boundary_seed(index));
    let (ha, wa) = if below(&mut rng, 5) == 0 {
        (60 + below(&mut rng, 200), 60 + below(&mut rng, 200))
    } else {
        (2 + below(&mut rng, 60), 2 + below(&mut rng, 60))
    };
    let (mut hb, mut wb) = (ha, wa);
    if below(&mut rng, 5) == 0 {
        hb = 2 + below(&mut rng, 60);
        wb = 2 + below(&mut rng, 60);
    }
    let a = nonempty_mask(&mut rng, ha, wa);
    let b = nonempty_mask(&mut rng, hb, wb);
    let r = 1 + below(&mut rng, 12) as i64;
    (
        SizedMask {
            h: ha,
            w: wa,
            bytes: a,
        },
        SizedMask {
            h: hb,
            w: wb,
            bytes: b,
        },
        r,
    )
}

/// One detection of a J&F sequence: track id and mask.
pub struct Detection {
    pub track: usize,
    pub mask: SizedMask,
}

/// `maskgen.jf_sequence`.
pub struct JfSequence {
    pub num_timesteps: usize,
    pub num_gt_ids: usize,
    pub num_pred_ids: usize,
    pub gt: Vec<Vec<Detection>>,
    pub pred: Vec<Vec<Detection>>,
}

pub fn jf_sequence(index: usize) -> JfSequence {
    let mut rng = SplitMix64::new(jf_sequence_seed(index));
    let big = below(&mut rng, 12) == 0;
    let (h, w, num_timesteps);
    if big {
        h = 400 + below(&mut rng, 800);
        w = 400 + below(&mut rng, 1200);
        num_timesteps = 1 + below(&mut rng, 3);
    } else {
        h = 2 + below(&mut rng, 20);
        w = 2 + below(&mut rng, 20);
        num_timesteps = if below(&mut rng, 6) == 0 {
            250 + below(&mut rng, 12)
        } else {
            1 + below(&mut rng, 25)
        };
    }
    let num_gt_ids = below(&mut rng, 5);
    let num_pred_ids = below(&mut rng, 6);
    let density = 1 + below(&mut rng, 4);
    let mut gt = Vec::with_capacity(num_timesteps);
    let mut pred = Vec::with_capacity(num_timesteps);
    for _ in 0..num_timesteps {
        for (side, count) in [(&mut gt, num_gt_ids), (&mut pred, num_pred_ids)] {
            let mut frame = Vec::new();
            for track in 0..count {
                if below(&mut rng, 4) < density {
                    let (mh, mw) = if below(&mut rng, 40) == 0 {
                        (h + 1, w)
                    } else {
                        (h, w)
                    };
                    let bytes = random_mask(&mut rng, mh, mw, !big);
                    frame.push(Detection {
                        track,
                        mask: SizedMask {
                            h: mh,
                            w: mw,
                            bytes,
                        },
                    });
                }
            }
            side.push(frame);
        }
    }
    JfSequence {
        num_timesteps,
        num_gt_ids,
        num_pred_ids,
        gt,
        pred,
    }
}
