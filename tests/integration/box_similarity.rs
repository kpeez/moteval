use crate::harness::testgen::{SplitMix64, fnv1a64, random_box};
use crate::harness::{
    as_array, as_u64, as_usize, assert_f64s_bit_equal, f64s_from_bits, read_json, testdata_dir,
};
use moteval::similarity::{box_ioa, box_iou};

#[test]
fn box_iou_and_ioa_match_python() {
    let json = read_json(&testdata_dir().join("box_similarity.json"));
    let cases = as_array(&json["cases"]);
    assert_eq!(cases.len(), 400);
    for case in cases {
        let seed = as_u64(&case["seed"]);
        let mut rng = SplitMix64::new(seed);
        let n = rng.below(7) as usize;
        let m = rng.below(7) as usize;
        let a: Vec<[f64; 4]> = (0..n).map(|_| random_box(&mut rng)).collect();
        let b: Vec<[f64; 4]> = (0..m).map(|_| random_box(&mut rng)).collect();
        assert_eq!((n, m), (as_usize(&case["n"]), as_usize(&case["m"])));
        let flat: Vec<f64> = a.iter().chain(&b).flatten().copied().collect();
        assert_eq!(
            fnv1a64(&flat),
            as_u64(&case["inputs_fnv"]),
            "seed {seed}: generator drift"
        );

        let iou = box_iou(&a, &b);
        assert_eq!(iou.shape(), (n, m));
        assert_f64s_bit_equal(
            &f64s_from_bits(&case["iou"]),
            iou.data(),
            &format!("iou seed {seed}"),
        );
        let ioa = box_ioa(&a, &b);
        assert_eq!(ioa.shape(), (n, m));
        assert_f64s_bit_equal(
            &f64s_from_bits(&case["ioa"]),
            ioa.data(),
            &format!("ioa seed {seed}"),
        );
    }
}
