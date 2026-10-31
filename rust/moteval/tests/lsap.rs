//! `moteval::lsap::linear_sum_assignment` against scipy 1.18.0 on 10,600 seeded matrices
//! and the special cases dumped by `rust/scripts/dump_lsap.py`. Index equality is exact.

mod harness;

use harness::testgen::{EPS, SplitMix64, fnv1a64};
use harness::{
    as_array, as_str, as_u64, as_usize, f64s_from_bits, read_json, testdata_dir, usizes,
};
use moteval::lsap::{LsapError, linear_sum_assignment};
use serde_json::Value as Json;

/// `(category, number of cases)`, as in `testgen.LSAP_CATEGORIES`.
const CATEGORIES: &[(&str, usize)] = &[
    ("square", 1500),
    ("wide", 1500),
    ("tall", 1500),
    ("integer_ties", 1500),
    ("clear_style", 1500),
    ("identity_style", 1500),
    ("hota_style", 1000),
    ("maximize", 500),
    ("large", 100),
];

fn lsap_seed(category: &str, index: usize) -> u64 {
    let base = CATEGORIES
        .iter()
        .position(|(name, _)| *name == category)
        .expect("known category");
    (base as u64 + 1) * 1_000_000 + index as u64
}

/// `testgen.lsap_case`: `(nr, nc, C-order cost, maximize)`.
fn lsap_case(category: &str, seed: u64) -> (usize, usize, Vec<f64>, bool) {
    let mut rng = SplitMix64::new(seed);
    let mut maximize = false;
    let below = |rng: &mut SplitMix64, n: u64| rng.below(n) as usize;
    let (nr, nc, cost) = match category {
        "square" => {
            let n = below(&mut rng, 13);
            (n, n, (0..n * n).map(|_| rng.uniform()).collect())
        }
        "wide" => {
            let nr = below(&mut rng, 10);
            let nc = nr + 1 + below(&mut rng, 6);
            (nr, nc, (0..nr * nc).map(|_| rng.uniform()).collect())
        }
        "tall" => {
            let nc = below(&mut rng, 10);
            let nr = nc + 1 + below(&mut rng, 6);
            (nr, nc, (0..nr * nc).map(|_| rng.uniform()).collect())
        }
        "integer_ties" => {
            let nr = 1 + below(&mut rng, 10);
            let nc = 1 + below(&mut rng, 10);
            (nr, nc, (0..nr * nc).map(|_| rng.below(4) as f64).collect())
        }
        "clear_style" => {
            let nr = 1 + below(&mut rng, 10);
            let nc = 1 + below(&mut rng, 10);
            let cost = (0..nr * nc)
                .map(|_| {
                    let mut sim = rng.uniform();
                    if rng.below(5) < 2 {
                        sim *= 0.5;
                    }
                    let continues = rng.below(8) == 0;
                    let score = if sim < 0.5 - EPS {
                        0.0
                    } else {
                        (if continues { 1000.0 } else { 0.0 }) + sim
                    };
                    -score
                })
                .collect();
            (nr, nc, cost)
        }
        "identity_style" => {
            let (cost, n) = identity_cost(&mut rng);
            (n, n, cost)
        }
        "hota_style" => {
            let nr = 1 + below(&mut rng, 10);
            let nc = 1 + below(&mut rng, 10);
            let cost = (0..nr * nc)
                .map(|_| {
                    let alignment = rng.uniform();
                    let sim = if rng.below(4) == 0 {
                        0.0
                    } else {
                        rng.uniform()
                    };
                    -(alignment * sim)
                })
                .collect();
            (nr, nc, cost)
        }
        "maximize" => {
            let nr = below(&mut rng, 9);
            let nc = below(&mut rng, 9);
            maximize = true;
            (
                nr,
                nc,
                (0..nr * nc).map(|_| rng.uniform() * 100.0).collect(),
            )
        }
        "large" => {
            let nr = 20 + below(&mut rng, 41);
            let nc = 20 + below(&mut rng, 41);
            let ties = rng.below(2) == 0;
            let cost = (0..nr * nc)
                .map(|_| {
                    if ties {
                        rng.below(6) as f64
                    } else {
                        rng.uniform()
                    }
                })
                .collect();
            (nr, nc, cost)
        }
        other => panic!("unknown category {other}"),
    };
    (nr, nc, cost, maximize)
}

/// `testgen._identity_cost`: Identity's `fn_mat + fp_mat` block matrix.
fn identity_cost(rng: &mut SplitMix64) -> (Vec<f64>, usize) {
    let num_gt = rng.below(7) as usize;
    let num_pred = rng.below(7) as usize;
    let gt_count: Vec<f64> = (0..num_gt).map(|_| (1 + rng.below(10)) as f64).collect();
    let pred_count: Vec<f64> = (0..num_pred).map(|_| (1 + rng.below(10)) as f64).collect();
    let mut matches = vec![vec![0.0; num_pred]; num_gt];
    for g in 0..num_gt {
        for p in 0..num_pred {
            if rng.below(2) != 0 {
                let most = gt_count[g].min(pred_count[p]) as u64;
                matches[g][p] = rng.below(most + 1) as f64;
            }
        }
    }
    let n = num_gt + num_pred;
    let mut fp = vec![vec![0.0; n]; n];
    let mut fn_ = vec![vec![0.0; n]; n];
    for row in fp.iter_mut().skip(num_gt) {
        for x in row.iter_mut().take(num_pred) {
            *x = 1e10;
        }
    }
    for row in fn_.iter_mut().take(num_gt) {
        for x in row.iter_mut().skip(num_pred) {
            *x = 1e10;
        }
    }
    for g in 0..num_gt {
        for x in fn_[g].iter_mut().take(num_pred) {
            *x = gt_count[g];
        }
        fn_[g][num_pred + g] = gt_count[g];
    }
    for p in 0..num_pred {
        for row in fp.iter_mut().take(num_gt) {
            row[p] = pred_count[p];
        }
        fp[p + num_gt][p] = pred_count[p];
    }
    for g in 0..num_gt {
        for p in 0..num_pred {
            fn_[g][p] -= matches[g][p];
            fp[g][p] -= matches[g][p];
        }
    }
    let cost = (0..n)
        .flat_map(|r| (0..n).map(move |c| (r, c)))
        .map(|(r, c)| fn_[r][c] + fp[r][c])
        .collect();
    (cost, n)
}

fn expected_error(message: &str) -> LsapError {
    match message {
        "matrix contains invalid numeric entries" => LsapError::InvalidEntries,
        "cost matrix is infeasible" => LsapError::Infeasible,
        other => panic!("unexpected scipy error {other:?}"),
    }
}

fn check(label: &str, case: &Json, nr: usize, nc: usize, cost: &[f64], maximize: bool) {
    let got = linear_sum_assignment(cost, nr, nc, maximize);
    match case.get("error") {
        Some(message) => {
            let want = expected_error(as_str(message));
            assert_eq!(got, Err(want), "{label}");
            assert_eq!(want.to_string(), as_str(message), "{label}: message");
        }
        None => {
            let (rows, cols) = got.unwrap_or_else(|err| panic!("{label}: {err}"));
            assert_eq!(rows, usizes(&case["rows"]), "{label}: rows");
            assert_eq!(cols, usizes(&case["cols"]), "{label}: cols");
        }
    }
}

#[test]
fn seeded_matrices_match_scipy() {
    let mut total = 0;
    for &(category, count) in CATEGORIES {
        let json = read_json(&testdata_dir().join("lsap").join(format!("{category}.json")));
        assert_eq!(as_str(&json["scipy"]), "1.18.0");
        let cases = as_array(&json["cases"]);
        assert_eq!(cases.len(), count, "{category}: case count");
        for case in cases {
            let index = as_usize(&case["i"]);
            let label = format!("{category}[{index}]");
            let (nr, nc, cost, maximize) = lsap_case(category, lsap_seed(category, index));
            assert_eq!(
                (nr, nc),
                (as_usize(&case["nr"]), as_usize(&case["nc"])),
                "{label}"
            );
            assert_eq!(
                fnv1a64(&cost),
                as_u64(&case["fnv"]),
                "{label}: generator drift"
            );
            check(&label, case, nr, nc, &cost, maximize);
            total += 1;
        }
    }
    assert_eq!(total, 10_600);
}

#[test]
fn special_matrices_match_scipy() {
    let json = read_json(&testdata_dir().join("lsap").join("special.json"));
    for case in as_array(&json["cases"]) {
        let label = as_str(&case["name"]);
        let cost = f64s_from_bits(&case["bits"]);
        let maximize = case["maximize"].as_bool().expect("bool");
        check(
            label,
            case,
            as_usize(&case["nr"]),
            as_usize(&case["nc"]),
            &cost,
            maximize,
        );
    }
}
