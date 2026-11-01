//! `moteval::numeric` against numpy 2.5.1 outputs dumped as bits by
//! `rust/scripts/dump_numeric.py`.

mod harness;

use harness::testgen::{fnv1a64, pattern};
use harness::{
    as_array, as_str, as_usize, assert_f64s_bit_equal, f64_from_bits, f64s_from_bits, read_json,
    same_bits, testdata_dir,
};
use moteval::numeric;
use serde_json::Value as Json;

fn numeric_json() -> Json {
    read_json(&testdata_dir().join("numeric.json"))
}

fn input(case: &Json, name: &str, n: usize, seed_field: &str) -> Vec<f64> {
    pattern(name, n, harness::as_u64(&case[seed_field]))
}

/// Regenerate a case's input and check it against the dumped fingerprint.
fn checked_input(case: &Json) -> Vec<f64> {
    let values = input(case, as_str(&case["pattern"]), as_usize(&case["n"]), "seed");
    assert_eq!(
        fnv1a64(&values),
        harness::as_u64(&case["inputs_fnv"]),
        "generator drift for {case}"
    );
    values
}

#[test]
fn sum_mean_and_cumsum_match_numpy() {
    let json = numeric_json();
    let cases = as_array(&json["reductions"]);
    assert!(cases.len() > 100);
    for case in cases {
        let a = checked_input(case);
        let label = format!("{} n={}", as_str(&case["pattern"]), a.len());
        assert_f64s_bit_equal(
            &[f64_from_bits(&case["sum"])],
            &[numeric::sum(&a)],
            &format!("sum {label}"),
        );
        if !case["mean"].is_null() {
            assert_f64s_bit_equal(
                &[f64_from_bits(&case["mean"])],
                &[numeric::mean(&a)],
                &format!("mean {label}"),
            );
        }
        if !case["cumsum"].is_null() {
            assert_f64s_bit_equal(
                &f64s_from_bits(&case["cumsum"]),
                &numeric::cumsum(&a),
                &format!("cumsum {label}"),
            );
        }
    }
}

#[test]
fn reduction_order_matters_on_these_inputs() {
    // The adversarial inputs must tell pairwise from sequential summation, or the test
    // above proves nothing about order.
    let json = numeric_json();
    let mut differs = 0;
    for case in as_array(&json["reductions"]) {
        let a = checked_input(case);
        let sequential = numeric::plain_sum(a.iter().copied());
        if !same_bits(sequential, f64_from_bits(&case["sum"])) {
            differs += 1;
        }
    }
    assert!(
        differs > 20,
        "only {differs} cases separate pairwise from sequential"
    );
}

#[test]
fn nanmean_matches_numpy() {
    let json = numeric_json();
    for case in as_array(&json["nanmean"]) {
        let a = checked_input(case);
        assert_f64s_bit_equal(
            &[f64_from_bits(&case["value"])],
            &[numeric::nanmean(&a)],
            &format!("nanmean {} n={}", as_str(&case["pattern"]), a.len()),
        );
    }
}

#[test]
fn axis_reductions_match_numpy() {
    let json = numeric_json();
    for case in as_array(&json["axes"]) {
        let (rows, cols) = (as_usize(&case["rows"]), as_usize(&case["cols"]));
        let a = input(case, as_str(&case["pattern"]), rows * cols, "seed");
        assert_eq!(fnv1a64(&a), harness::as_u64(&case["inputs_fnv"]));
        let label = format!("{} {rows}x{cols}", as_str(&case["pattern"]));
        assert_f64s_bit_equal(
            &[f64_from_bits(&case["sum_all"])],
            &[numeric::sum(&a)],
            &format!("sum_all {label}"),
        );
        assert_f64s_bit_equal(
            &f64s_from_bits(&case["sum_axis0"]),
            &numeric::sum_axis0(&a, rows, cols),
            &format!("sum_axis0 {label}"),
        );
        assert_f64s_bit_equal(
            &f64s_from_bits(&case["mean_axis0"]),
            &numeric::mean_axis0(&a, rows, cols),
            &format!("mean_axis0 {label}"),
        );
        assert_f64s_bit_equal(
            &f64s_from_bits(&case["sum_last_axis"]),
            &numeric::sum_last_axis(&a, cols),
            &format!("sum_last_axis {label}"),
        );
        assert_f64s_bit_equal(
            &f64s_from_bits(&case["mean_last_axis"]),
            &numeric::mean_last_axis(&a, cols),
            &format!("mean_last_axis {label}"),
        );
    }
}

#[test]
fn average_matches_numpy() {
    let json = numeric_json();
    for case in as_array(&json["average"]) {
        let n = as_usize(&case["n"]);
        let a = pattern("spread", n, harness::as_u64(&case["seed"]));
        let w = pattern("uniform", n, harness::as_u64(&case["weights_seed"]));
        let both: Vec<f64> = a.iter().chain(&w).copied().collect();
        assert_eq!(fnv1a64(&both), harness::as_u64(&case["inputs_fnv"]));
        assert_f64s_bit_equal(
            &[f64_from_bits(&case["value"])],
            &[numeric::average(&a, &w).expect("weights sum to nonzero")],
            &format!("average n={n}"),
        );
    }
    assert_eq!(
        numeric::average(&[1.0], &[0.0]),
        Err(numeric::NumericError::WeightsSumToZero)
    );
    assert_eq!(
        numeric::NumericError::WeightsSumToZero.to_string(),
        "Weights sum to zero, can't be normalized"
    );
}

#[test]
fn plain_sum_matches_python_reduce() {
    let json = numeric_json();
    for case in as_array(&json["plain_sum"]) {
        let a = checked_input(case);
        assert_f64s_bit_equal(
            &[f64_from_bits(&case["value"])],
            &[numeric::plain_sum(a.iter().copied())],
            &format!("plain_sum {} n={}", as_str(&case["pattern"]), a.len()),
        );
    }
}

#[test]
fn arange_matches_numpy() {
    let json = numeric_json();
    for case in as_array(&json["arange"]) {
        let (start, stop, step) = (
            f64_from_bits(&case["start"]),
            f64_from_bits(&case["stop"]),
            f64_from_bits(&case["step"]),
        );
        assert_f64s_bit_equal(
            &f64s_from_bits(&case["out"]),
            &numeric::arange(start, stop, step).expect("finite arange"),
            &format!("arange({start}, {stop}, {step})"),
        );
    }
    assert_eq!(
        numeric::arange(0.0, f64::NAN, 1.0),
        Err(numeric::NumericError::ArangeLength)
    );
    assert_eq!(
        numeric::arange(0.0, 1.0, 0.0),
        Err(numeric::NumericError::ArangeZeroStep)
    );
}

#[test]
fn linspace_matches_numpy() {
    let json = numeric_json();
    let cases = as_array(&json["linspace"]);
    assert!(cases.len() > 600);
    for case in cases {
        let (start, stop) = (f64_from_bits(&case["start"]), f64_from_bits(&case["stop"]));
        let num = as_usize(&case["num"]);
        let endpoint = case["endpoint"].as_bool().expect("bool");
        assert_f64s_bit_equal(
            &f64s_from_bits(&case["out"]),
            &numeric::linspace(start, stop, num, endpoint),
            &format!("linspace({start}, {stop}, {num}, endpoint={endpoint})"),
        );
    }
}

/// The SVML `log10` port against numpy's SVML path, on every host: small integers, every
/// binade, mantissas where the rounded `vrcp14pd` switches (13 of them change the result
/// if `1/m` stands in for `vrcp14pd`), subnormals, and a fingerprint of
/// `np.log10(1..=1_000_000)`.
#[test]
fn log10_svml_matches_numpy() {
    let json = numeric_json();
    let case = &json["log10"];
    assert_eq!(as_str(&case["kernel"]), "svml");
    let xs = f64s_from_bits(&case["x"]);
    let got: Vec<f64> = xs.iter().map(|&x| numeric::log10_svml(x)).collect();
    assert_f64s_bit_equal(&f64s_from_bits(&case["out"]), &got, "log10_svml");
    // The fixture tells SVML from libm only if they disagree somewhere in it. That was
    // measured for glibc (397 inputs); other libms are not asserted on.
    let libm_differs = xs
        .iter()
        .zip(&got)
        .filter(|&(&x, &y)| !same_bits(x.log10(), y))
        .count();
    if cfg!(all(target_os = "linux", target_env = "gnu")) {
        assert!(
            libm_differs > 0,
            "no input where glibc log10 differs from SVML"
        );
    }
    let n = harness::as_u64(&case["integers"]);
    let ints: Vec<f64> = (1..=n).map(|i| numeric::log10_svml(i as f64)).collect();
    assert_eq!(
        fnv1a64(&ints),
        harness::as_u64(&case["integers_fnv"]),
        "log10_svml(1..={n}) differs from numpy"
    );
    eprintln!(
        "log10: {} inputs ({libm_differs} where libm differs) and 1..={n}; numpy on this host \
         uses {}",
        xs.len(),
        if numeric::numpy_uses_svml() {
            "SVML"
        } else {
            "libm"
        }
    );
}

#[test]
fn round_matches_numpy() {
    let json = numeric_json();
    for case in as_array(&json["round"]) {
        let x = f64_from_bits(&case["x"]);
        assert_f64s_bit_equal(
            &[f64_from_bits(&case["out"])],
            &[numeric::round_half_even(x)],
            &format!("round({x:e})"),
        );
    }
}

#[test]
fn extrema_match_numpy_and_python() {
    let json = numeric_json();
    for case in as_array(&json["extrema"]) {
        let (a, b) = (f64_from_bits(&case["a"]), f64_from_bits(&case["b"]));
        let label = format!("({a:e}, {b:e})");
        for (field, got) in [
            ("maximum", numeric::maximum(a, b)),
            ("minimum", numeric::minimum(a, b)),
            ("py_max", numeric::py_max(a, b)),
            ("py_min", numeric::py_min(a, b)),
        ] {
            assert_f64s_bit_equal(
                &[f64_from_bits(&case[field])],
                &[got],
                &format!("{field}{label}"),
            );
        }
    }
}

#[test]
fn hota_and_trackmap_constants_match_numpy() {
    // Spot-check the three constant ranges the metrics build.
    let json = numeric_json();
    let aranges = as_array(&json["arange"]);
    let alphas = numeric::arange(0.05, 0.99, 0.05).expect("alphas");
    assert_eq!(alphas.len(), 19);
    assert_f64s_bit_equal(&f64s_from_bits(&aranges[0]["out"]), &alphas, "HOTA alphas");
    let iou = numeric::arange(0.5, 0.96, 0.05).expect("iou thresholds");
    assert_eq!(iou.len(), 10);
    assert_f64s_bit_equal(&f64s_from_bits(&aranges[1]["out"]), &iou, "TrackMAP IoU");
    let recall = numeric::linspace(0.0, 1.0, 101, true);
    let linspaces = as_array(&json["linspace"]);
    assert_f64s_bit_equal(&f64s_from_bits(&linspaces[0]["out"]), &recall, "recall");
}
