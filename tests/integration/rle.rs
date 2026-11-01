//! `moteval::rle` and the mask functions of `moteval::similarity` against pycocotools
//! 2.0.11 and Python moteval on 1,000 seeded masks (`tests/integration/data/rle.json` and
//! `rle_iou.json`, from `scripts/testdata/dump_masks.py`). Every output must be equal: strings
//! byte for byte, floats bit for bit.

use crate::harness::testgen::SplitMix64;
use crate::harness::{as_array, as_str, as_u64, as_usize, assert_f64s_bit_equal, f64s_from_bits};
use crate::harness::{read_json, testdata_dir};
use crate::maskgen::{FAMILIES, FAMILY_SIZE, family_seed, fnv1a64_bytes, mask_size, random_mask};
use moteval::rle::{self, RleError, RleMask};
use moteval::similarity::{mask_ioa, mask_iou, masks_overlap, merge_masks};
use serde_json::Value as Json;

/// A mask from a dumped `{"h", "w", "counts"}` record.
fn rle_of(json: &Json) -> RleMask {
    RleMask::from_compressed(
        as_usize(&json["h"]),
        as_usize(&json["w"]),
        as_str(&json["counts"]).as_bytes(),
    )
    .expect("dumped RLE is valid")
}

fn assert_same_rle(want: &Json, got: &RleMask, label: &str) {
    assert_eq!(
        (got.height(), got.width()),
        (as_usize(&want["h"]), as_usize(&want["w"])),
        "{label}: size"
    );
    assert_eq!(
        String::from_utf8(got.to_compressed()).expect("ascii"),
        as_str(&want["counts"]),
        "{label}: counts"
    );
}

fn crowd_flags(json: &Json) -> Vec<bool> {
    as_array(json).iter().map(|v| as_u64(v) == 1).collect()
}

#[test]
fn pycocotools_fuzz_matches_on_1000_masks() {
    let json = read_json(&testdata_dir().join("rle.json"));
    let ious = read_json(&testdata_dir().join("rle_iou.json"));
    let families = as_array(&json["families"]);
    let family_ious = as_array(&ious["families"]);
    assert_eq!(family_ious.len(), FAMILIES);
    assert_eq!(families.len(), FAMILIES);
    let mut all: Vec<Vec<RleMask>> = Vec::with_capacity(FAMILIES);
    let mut masks_checked = 0;
    for (f, family) in families.iter().enumerate() {
        let label = format!("family {f}");
        let family_iou = &family_ious[f];
        assert_eq!(as_u64(&family["seed"]), family_seed(f));
        let mut rng = SplitMix64::new(family_seed(f));
        let (h, w) = mask_size(&mut rng);
        assert_eq!((h, w), (as_usize(&family["h"]), as_usize(&family["w"])));
        let masks: Vec<Vec<u8>> = (0..FAMILY_SIZE)
            .map(|_| random_mask(&mut rng, h, w, true))
            .collect();
        let split = 1 + rng.below(FAMILY_SIZE as u64 - 1) as usize;
        let crowd: Vec<bool> = (0..FAMILY_SIZE - split)
            .map(|_| rng.below(2) == 1)
            .collect();
        assert_eq!(
            split,
            as_usize(&family["split"]),
            "{label}: generator drift"
        );
        assert_eq!(
            crowd,
            crowd_flags(&family["crowd"]),
            "{label}: generator drift"
        );

        let mut rles = Vec::with_capacity(FAMILY_SIZE);
        for (k, (mask, want)) in masks.iter().zip(as_array(&family["masks"])).enumerate() {
            let label = format!("{label} mask {k}");
            assert_eq!(
                fnv1a64_bytes(mask),
                as_u64(&want["fnv"]),
                "{label}: generator drift"
            );
            let counts = as_str(&want["counts"]);
            let encoded = rle::encode(mask, h, w);
            assert_eq!(
                String::from_utf8(encoded.to_compressed()).expect("ascii"),
                counts,
                "{label}: encode"
            );
            let python = RleMask::from_compressed(h, w, counts.as_bytes()).expect("valid");
            assert_eq!(python, encoded, "{label}: parsed string");
            assert_eq!(&rle::decode(&python), mask, "{label}: decode");
            assert_eq!(rle::area(&python), as_u64(&want["area"]), "{label}: area");
            assert_f64s_bit_equal(
                &f64s_from_bits(&want["bbox"]),
                &rle::to_bbox(&python),
                &format!("{label}: toBbox"),
            );
            let one = rle::merge(std::slice::from_ref(&python), false).expect("merge");
            assert_eq!(
                String::from_utf8(one.to_compressed()).expect("ascii"),
                as_str(&want["merge_one"]),
                "{label}: string round trip"
            );
            rles.push(python);
            masks_checked += 1;
        }

        let check_iou = |dt: &[RleMask], gt: &[RleMask], crowd: &[bool], key: &str| {
            let got = rle::iou(dt, gt, crowd).expect("iou");
            assert_f64s_bit_equal(
                &f64s_from_bits(&family_iou[key]),
                &got,
                &format!("{label} {key}"),
            );
        };
        check_iou(&rles, &rles, &[false; FAMILY_SIZE], "iou_plain");
        check_iou(&rles, &rles, &[true; FAMILY_SIZE], "iou_crowd");
        check_iou(&rles[..split], &rles[split..], &crowd, "iou_split");
        let plain = mask_iou(&rles, &rles).expect("mask_iou");
        assert_f64s_bit_equal(
            &f64s_from_bits(&family_iou["iou_plain"]),
            plain.data(),
            &format!("{label} mask_iou"),
        );
        let ioa = mask_ioa(&rles, &rles).expect("mask_ioa");
        assert_f64s_bit_equal(
            &f64s_from_bits(&family_iou["iou_crowd"]),
            ioa.data(),
            &format!("{label} mask_ioa"),
        );
        assert_same_rle(
            &family["merge_union"],
            &rle::merge(&rles, false).unwrap(),
            &format!("{label} merge union"),
        );
        assert_same_rle(
            &family["merge_union"],
            &merge_masks(&rles).unwrap(),
            &format!("{label} merge_masks"),
        );
        assert_same_rle(
            &family["merge_intersect"],
            &rle::merge(&rles, true).unwrap(),
            &format!("{label} merge intersect"),
        );
        assert_same_rle(
            &family["pair_union"],
            &rle::merge(&rles[..2], false).unwrap(),
            &format!("{label} pair union"),
        );
        assert_same_rle(
            &family["pair_intersect"],
            &rle::merge(&rles[..2], true).unwrap(),
            &format!("{label} pair intersect"),
        );
        assert_eq!(
            masks_overlap(&rles).unwrap(),
            family["overlap"].as_bool().unwrap(),
            "{label}: masks_overlap"
        );
        assert_eq!(
            masks_overlap(&rles[..2]).unwrap(),
            family["pair_overlap"].as_bool().unwrap(),
            "{label}: pair masks_overlap"
        );
        all.push(rles);
    }
    assert_eq!(masks_checked, 1000);

    let cross = as_array(&json["cross"]);
    let cross_ious = as_array(&ious["cross"]);
    assert_eq!(cross_ious.len(), FAMILIES - 1);
    assert_eq!(cross.len(), FAMILIES - 1);
    for (f, case) in cross.iter().enumerate() {
        let (a, b) = (&all[f], &all[f + 1]);
        let label = format!("cross {f}");
        let iou = rle::iou(a, b, &[false; FAMILY_SIZE]).unwrap();
        assert_f64s_bit_equal(&f64s_from_bits(&cross_ious[f]["iou"]), &iou, &label);
        let ioa = mask_ioa(a, b).unwrap();
        assert_f64s_bit_equal(&f64s_from_bits(&cross_ious[f]["ioa"]), ioa.data(), &label);
        let pair = [a[0].clone(), b[0].clone()];
        assert_same_rle(&case["merge"], &rle::merge(&pair, false).unwrap(), &label);
        assert_eq!(
            masks_overlap(&pair).unwrap(),
            case["overlap"].as_bool().unwrap(),
            "{label}: masks_overlap"
        );
    }

    // Non-canonical strings parse to the same runs; Python keeps the string as given.
    let non_canonical = as_array(&json["non_canonical"]);
    assert_eq!(non_canonical.len(), 100);
    for (k, case) in non_canonical.iter().enumerate() {
        let label = format!("non-canonical {k}");
        let mask = rle_of(case);
        assert_eq!(
            String::from_utf8(mask.to_compressed()).unwrap(),
            as_str(&case["counts"]),
            "{label}: counts kept as given"
        );
        assert_eq!(
            fnv1a64_bytes(&rle::decode(&mask)),
            as_u64(&case["fnv"]),
            "{label}: decode"
        );
        assert_eq!(rle::area(&mask), as_u64(&case["area"]), "{label}: area");
        assert_f64s_bit_equal(&f64s_from_bits(&case["bbox"]), &rle::to_bbox(&mask), &label);
        let one = rle::merge(std::slice::from_ref(&mask), false).unwrap();
        assert_eq!(
            String::from_utf8(one.to_compressed()).unwrap(),
            as_str(&case["merge_one"]),
            "{label}: re-encoded"
        );
    }

    // pycocotools' decode rejects runs that overrun the mask; construction rejects them
    // with the same message.
    let overrun = &json["overrun"];
    let err = RleMask::from_compressed(
        as_usize(&overrun["h"]),
        as_usize(&overrun["w"]),
        as_str(&overrun["counts"]).as_bytes(),
    )
    .unwrap_err();
    assert_eq!(err, RleError::InvalidCounts);
    assert_eq!(err.to_string(), as_str(&overrun["error"]));
}

#[test]
fn empty_inputs_keep_their_shape() {
    let a = vec![rle::encode(&[0, 1, 1, 0], 2, 2)];
    assert_eq!(mask_iou(&a, &[]).unwrap().shape(), (1, 0));
    assert_eq!(mask_ioa(&[], &a).unwrap().shape(), (0, 1));
    assert!(rle::iou(&a, &a, &[]).is_err());
    assert!(!masks_overlap(&[]).unwrap());
}
