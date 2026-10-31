//! TrackMAP's frame-union set order: `union_frame_order` equals the order CPython 3.14.8
//! iterates `set(gt.keys()) | set(dt.keys())`, on the seeded key sets and insertion
//! orders `rust/scripts/dump_set_order.py` dumped through moteval's own `_union_frames`.
//!
//! This file regenerates every case from its seed, draw for draw the same as the Python
//! generator, and checks `inputs_fnv` first so a generator mismatch fails as such.

mod harness;

use std::collections::HashSet;

use harness::testgen::SplitMix64;
use harness::{as_array, as_str, as_u64, as_usize, read_json, testdata_dir};
use moteval::metrics::track_map::union_frame_order;

const SEPARATOR: u64 = u64::MAX;
const SPANS: [u64; 5] = [16, 64, 256, 4096, 100_000];

/// FNV-1a over each key's little-endian `u64` bytes (`fnv1a64_keys`).
fn fnv1a64_keys(keys: impl IntoIterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for key in keys {
        for byte in key.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

/// `n` distinct keys below `span`, in draw order.
fn distinct(rng: &mut SplitMix64, n: u64, span: u64) -> Vec<u64> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    while (out.len() as u64) < n {
        let key = rng.below(span);
        if seen.insert(key) {
            out.push(key);
        }
    }
    out
}

/// Fisher-Yates, from the last position down.
fn shuffle(rng: &mut SplitMix64, keys: &mut [u64]) {
    for i in (1..keys.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        keys.swap(i, j);
    }
}

/// `set_order_case(regime, seed)`: `(gt_keys, dt_keys)` in dict insertion order.
fn set_order_case(regime: &str, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let mut rng = SplitMix64::new(seed);
    match regime {
        "dense_ascending" => {
            let g_start = rng.below(3000);
            let g_len = rng.below(200);
            let d_start = (g_start + rng.below(g_len + 41)).saturating_sub(20);
            let d_len = rng.below(200);
            (
                (g_start..g_start + g_len).collect(),
                (d_start..d_start + d_len).collect(),
            )
        }
        "sparse_ascending" | "sparse_shuffled" => {
            let span = SPANS[rng.below(SPANS.len() as u64) as usize];
            let cap = (span / 2).min(300);
            let n = rng.below(cap + 1);
            let mut gt = distinct(&mut rng, n, span);
            let n = rng.below(cap + 1);
            let mut dt = distinct(&mut rng, n, span);
            if regime == "sparse_ascending" {
                gt.sort_unstable();
                dt.sort_unstable();
            }
            (gt, dt)
        }
        "colliding" => {
            let base = 1u64 << (3 + rng.below(10));
            let mut sides = Vec::new();
            for _ in 0..2 {
                let n = rng.below(40) as usize;
                let mut seen = HashSet::new();
                let mut keys = Vec::new();
                while keys.len() < n {
                    let offset = rng.below(3);
                    let key = offset + base * rng.below(64);
                    if seen.insert(key) {
                        keys.push(key);
                    }
                }
                if rng.below(2) == 1 {
                    keys.sort_unstable();
                }
                sides.push(keys);
            }
            let dt = sides.pop().expect("two sides");
            let gt = sides.pop().expect("two sides");
            (gt, dt)
        }
        "one_side_empty" => {
            let span = SPANS[rng.below(SPANS.len() as u64) as usize];
            let cap = (span / 2).min(300);
            let n = 1 + rng.below(cap);
            let keys = distinct(&mut rng, n, span);
            match rng.below(3) {
                0 => (keys, Vec::new()),
                1 => (Vec::new(), keys),
                _ => (Vec::new(), Vec::new()),
            }
        }
        "subset_overlap" => {
            let span = SPANS[rng.below(SPANS.len() as u64) as usize];
            let cap = (span / 2).min(300);
            let n = rng.below(cap + 1);
            let gt = distinct(&mut rng, n, span);
            let mut dt: Vec<u64> = gt.iter().copied().filter(|_| rng.below(2) == 0).collect();
            shuffle(&mut rng, &mut dt);
            let mut dt_seen: HashSet<u64> = dt.iter().copied().collect();
            for _ in 0..rng.below(10) {
                let key = rng.below(span);
                if dt_seen.insert(key) {
                    dt.push(key);
                }
            }
            (gt, dt)
        }
        "short_far_tracks" => {
            let num_frames = 100 + rng.below(5000);
            let g_start = rng.below(num_frames);
            let g_len = 1 + rng.below(8);
            let d_start = (g_start + rng.below(17)).saturating_sub(8);
            let d_len = 1 + rng.below(8);
            (
                (g_start..(g_start + g_len).min(num_frames)).collect(),
                (d_start..d_start + d_len).collect(),
            )
        }
        "huge" => {
            let n = 50_000 + rng.below(20_000);
            let mut gt = distinct(&mut rng, n, 1_000_000);
            if rng.below(2) == 1 {
                gt.sort_unstable();
            }
            let n = 1 + rng.below(5000);
            let dt = distinct(&mut rng, n, 1_000_000);
            (gt, dt)
        }
        other => panic!("unknown regime {other:?}"),
    }
}

#[test]
fn union_frame_order_matches_cpython_set_iteration() {
    let json = read_json(&testdata_dir().join("set_order.json"));
    assert_eq!(as_u64(&json["format"]), 1);
    assert_eq!(as_str(&json["python"]), "3.14.8");
    let mut cases = 0;
    let mut non_ascending = 0;
    for regime in as_array(&json["regimes"]) {
        let name = as_str(&regime["regime"]);
        let first_seed = as_u64(&regime["first_seed"]);
        for (i, row) in as_array(&regime["cases"]).iter().enumerate() {
            let seed = first_seed + i as u64;
            let (gt, dt) = set_order_case(name, seed);
            let inputs = gt
                .iter()
                .copied()
                .chain([SEPARATOR])
                .chain(dt.iter().copied());
            assert_eq!(
                fnv1a64_keys(inputs),
                as_u64(&row[0]),
                "{name} seed {seed}: generator drift"
            );
            let order = union_frame_order(&dt, &gt);
            assert_eq!(order.len(), as_usize(&row[1]), "{name} seed {seed}: length");
            assert_eq!(
                fnv1a64_keys(order.iter().copied()),
                as_u64(&row[2]),
                "{name} seed {seed}: order differs from CPython: {order:?}"
            );
            let ascending = order.windows(2).all(|w| w[1] > w[0]);
            assert_eq!(u64::from(ascending), as_u64(&row[3]));
            non_ascending += usize::from(!ascending);
            cases += 1;
        }
    }
    assert_eq!(cases, 4304);
    assert_eq!(non_ascending, as_usize(&json["non_ascending"]));
    // The fuzz must reach the orders the fallback path exists for.
    assert!(
        non_ascending > 1000,
        "only {non_ascending} non-ascending cases"
    );
}
