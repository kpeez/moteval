//! `pyfmt` against strings CPython 3.14.8 wrote (`scripts/testdata/dump_pyfmt.py`):
//! `repr(float)` and `format(x, "1.5g")` on 100,000 seeded doubles plus edge cases, and
//! the JSON string, CSV row, and path forms.

use crate::harness::testgen::{SplitMix64, fnv1a64, pow2};
use crate::harness::{as_array, as_str, as_u64, as_usize, read_json, testdata_dir};
use moteval::pyfmt::{csv_row, format_g, json_float, json_str, path_str, repr_float};
use serde_json::Value as Json;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn data() -> Json {
    read_json(&testdata_dir().join("pyfmt.json"))
}

fn fnv_text(texts: &[String]) -> u64 {
    let mut h = FNV_OFFSET;
    for text in texts {
        for &byte in text.as_bytes().iter().chain(b"\n") {
            h = (h ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
    }
    h
}

// The generators of dump_pyfmt.py, draw for draw.
fn gen_bits(rng: &mut SplitMix64) -> f64 {
    f64::from_bits(rng.next_u64())
}

fn gen_decimal(rng: &mut SplitMix64) -> f64 {
    let ndigits = 1 + rng.below(17) as u32;
    let mantissa = rng.below(10u64.pow(ndigits));
    let exponent = rng.below(61) as i64 - 30;
    let value: f64 = format!("{mantissa}e{exponent}").parse().unwrap();
    if rng.next_u64() & 1 == 1 {
        -value
    } else {
        value
    }
}

fn gen_ratio(rng: &mut SplitMix64) -> f64 {
    let n = 1 + rng.below(10000);
    let k = rng.below(n + 1);
    let ratio = k as f64 / n as f64;
    if rng.below(2) == 0 {
        ratio
    } else {
        100.0 * ratio
    }
}

fn gen_tie5(rng: &mut SplitMix64) -> f64 {
    let mantissa = (10000 + rng.below(90000)) * 10 + 5;
    let exponent = rng.below(41) as i64 - 20;
    format!("{mantissa}e{exponent}").parse().unwrap()
}

fn gen_dyadic(rng: &mut SplitMix64) -> f64 {
    let mantissa = rng.below(1 << 24);
    let shift = rng.below(40) as i32;
    mantissa as f64 * pow2(-shift)
}

fn gen_wide_dyadic(rng: &mut SplitMix64) -> f64 {
    let mantissa = (1u64 << 52) + rng.below(1 << 52);
    let shift = rng.below(12) as i32;
    mantissa as f64 * pow2(-shift)
}

fn generator(name: &str) -> fn(&mut SplitMix64) -> f64 {
    match name {
        "bits" => gen_bits,
        "decimal" => gen_decimal,
        "ratio" => gen_ratio,
        "tie5" => gen_tie5,
        "dyadic" => gen_dyadic,
        "wide_dyadic" => gen_wide_dyadic,
        other => panic!("unknown seeded set {other:?}"),
    }
}

/// Compare one output kind chunk by chunk; on a mismatch, print the chunk in full.
fn check_chunks(set: &str, kind: &str, values: &[f64], out: &[String], want: &Json, chunk: usize) {
    let want = as_array(want);
    assert_eq!(
        want.len(),
        values.len().div_ceil(chunk),
        "{set}/{kind}: chunk count"
    );
    for (index, expected) in want.iter().enumerate() {
        let start = index * chunk;
        let end = (start + chunk).min(values.len());
        if fnv_text(&out[start..end]) != as_u64(expected) {
            let shown: Vec<String> = (start..end)
                .map(|i| format!("{:#018x} -> {}", values[i].to_bits(), out[i]))
                .collect();
            panic!(
                "{set}/{kind}: chunk {index} (values {start}..{end}) differs from Python; \
                 Rust wrote:\n{}",
                shown.join("\n")
            );
        }
    }
}

#[test]
fn repr_and_g5_match_python_on_seeded_doubles() {
    let data = data();
    let chunk = as_usize(&data["chunk"]);
    let mut total = 0;
    for set in as_array(&data["sets"]) {
        let name = as_str(&set["name"]);
        let n = as_usize(&set["n"]);
        let gen_one = generator(name);
        let mut rng = SplitMix64::new(as_u64(&set["seed"]));
        let values: Vec<f64> = (0..n).map(|_| gen_one(&mut rng)).collect();
        assert_eq!(
            fnv1a64(&values),
            as_u64(&set["inputs_fnv"]),
            "{name}: generator drift (inputs differ from Python's)"
        );
        let reprs: Vec<String> = values.iter().map(|&x| repr_float(x)).collect();
        let g5: Vec<String> = values.iter().map(|&x| format_g(x, 5)).collect();
        check_chunks(name, "repr", &values, &reprs, &set["repr_chunks"], chunk);
        check_chunks(name, "g5", &values, &g5, &set["g5_chunks"], chunk);
        total += n;
    }
    assert!(total >= 100_000, "only {total} seeded doubles");
    println!("pyfmt: {total} seeded doubles match repr() and format(x, '1.5g')");
}

#[test]
fn float_edge_cases_match_python() {
    let data = data();
    let cases = as_array(&data["edge"]);
    for case in cases {
        let x = f64::from_bits(as_u64(&case["bits"]));
        let label = format!("{:#018x} ({})", x.to_bits(), as_str(&case["repr"]));
        assert_eq!(repr_float(x), as_str(&case["repr"]), "repr {label}");
        assert_eq!(repr_float(x), as_str(&case["str"]), "str {label}");
        assert_eq!(format_g(x, 5), as_str(&case["g5"]), "1.5g {label}");
        assert_eq!(json_float(x), as_str(&case["json"]), "json {label}");
        let list = format!("[{}, {}]", json_float(x), json_float(x));
        assert_eq!(list, as_str(&case["json_list"]), "json list {label}");
    }
    assert!(cases.len() > 150, "only {} edge cases", cases.len());
}

#[test]
fn json_strings_match_python() {
    for case in as_array(&data()["json_strings"]) {
        let text = as_str(&case["text"]);
        assert_eq!(json_str(text), as_str(&case["json"]), "{text:?}");
    }
}

#[test]
fn csv_rows_match_python() {
    for case in as_array(&data()["csv_rows"]) {
        let row: Vec<&str> = as_array(&case["row"]).iter().map(as_str).collect();
        assert_eq!(csv_row(&row), as_str(&case["csv"]), "{row:?}");
    }
}

#[test]
fn paths_match_python() {
    for case in as_array(&data()["paths"]) {
        let text = as_str(&case["text"]);
        assert_eq!(path_str(text), as_str(&case["str"]), "{text:?}");
    }
}
