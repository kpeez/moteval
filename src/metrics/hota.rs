//! HOTA metric family over 19 alpha thresholds, plus HOTA(0), LocA(0), HOTALocA(0),
//! ported from `moteval/metrics/hota.py` with all three combiners.
//!
//! TrackEval's global alignment (a Hungarian assignment weighted by a Jaccard-normalized
//! potential-match count) runs once per frame; the eps-guarded `similarity >= alpha - eps`
//! test then scores the 19 alphas. Every sum follows the reduction order of its Python
//! line (see the comments that quote it).

use crate::lsap::linear_sum_assignment;
use crate::metrics::{
    Metric, MetricError, Scores, Value, combine_sum, combine_weighted_av, require_all,
};
use crate::model::SequenceData;
use crate::numeric::{self, EPS, maximum};

/// The public fields, in the Python `fields` order.
pub const FIELDS: &[&str] = &[
    "HOTA",
    "DetA",
    "AssA",
    "DetRe",
    "DetPr",
    "AssRe",
    "AssPr",
    "LocA",
    "OWTA",
    "HOTA_TP",
    "HOTA_FN",
    "HOTA_FP",
    "HOTA(0)",
    "LocA(0)",
    "HOTALocA(0)",
];

/// `_FLOAT_ARRAY_FIELDS`: the 19-alpha float arrays.
const FLOAT_ARRAY_FIELDS: [&str; 9] = [
    "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA",
];
/// `_INTEGER_ARRAY_FIELDS`: the 19-alpha counts (held as floats, as in Python).
const INTEGER_ARRAY_FIELDS: [&str; 3] = ["HOTA_TP", "HOTA_FN", "HOTA_FP"];
/// `_FLOAT_FIELDS`: the alpha-0 scalars.
const FLOAT_FIELDS: [&str; 3] = ["HOTA(0)", "LocA(0)", "HOTALocA(0)"];

/// The number of alphas, `len(np.arange(0.05, 0.99, 0.05))`.
pub const NUM_ALPHAS: usize = 19;

type Alphas = [f64; NUM_ALPHAS];

/// `ALPHAS = np.arange(0.05, 0.99, 0.05)`, built with numpy's `arange` formula.
pub fn alphas() -> Alphas {
    let values = numeric::arange(0.05, 0.99, 0.05).expect("a finite, nonzero-step range");
    values
        .try_into()
        .expect("np.arange(0.05, 0.99, 0.05) has 19 elements")
}

/// `moteval.metrics.hota.HOTA`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hota;

/// `arrays[name]` as a 19-element array.
fn alpha_array(scores: &Scores, field: &str) -> Alphas {
    scores
        .array(field)
        .try_into()
        .unwrap_or_else(|_| panic!("HOTA field {field:?} does not hold {NUM_ALPHAS} values"))
}

/// `_compute_det_fields`: DetRe, DetPr, DetA, HOTA, OWTA, inserted in that order.
fn compute_det_fields(arrays: &mut Scores) {
    let tp = alpha_array(arrays, "HOTA_TP");
    let fn_ = alpha_array(arrays, "HOTA_FN");
    let fp = alpha_array(arrays, "HOTA_FP");
    let ass_a = alpha_array(arrays, "AssA");
    let mut det_re = [0.0; NUM_ALPHAS];
    let mut det_pr = [0.0; NUM_ALPHAS];
    let mut det_a = [0.0; NUM_ALPHAS];
    let mut hota = [0.0; NUM_ALPHAS];
    let mut owta = [0.0; NUM_ALPHAS];
    for a in 0..NUM_ALPHAS {
        // HOTA_TP / np.maximum(1, HOTA_TP + HOTA_FN)
        det_re[a] = tp[a] / maximum(1.0, tp[a] + fn_[a]);
        // HOTA_TP / np.maximum(1, HOTA_TP + HOTA_FP)
        det_pr[a] = tp[a] / maximum(1.0, tp[a] + fp[a]);
        // HOTA_TP / np.maximum(1, HOTA_TP + HOTA_FN + HOTA_FP)
        det_a[a] = tp[a] / maximum(1.0, (tp[a] + fn_[a]) + fp[a]);
        // np.sqrt(DetA * AssA), np.sqrt(DetRe * AssA)
        hota[a] = (det_a[a] * ass_a[a]).sqrt();
        owta[a] = (det_re[a] * ass_a[a]).sqrt();
    }
    arrays.insert("DetRe", Value::Array(det_re.to_vec()));
    arrays.insert("DetPr", Value::Array(det_pr.to_vec()));
    arrays.insert("DetA", Value::Array(det_a.to_vec()));
    arrays.insert("HOTA", Value::Array(hota.to_vec()));
    arrays.insert("OWTA", Value::Array(owta.to_vec()));
}

/// `_finalize`: append HOTA(0), LocA(0), and their product.
fn finalize(mut arrays: Scores) -> Scores {
    let hota_0 = arrays.array("HOTA")[0];
    let loca_0 = arrays.array("LocA")[0];
    arrays.insert("HOTA(0)", Value::Float(hota_0));
    arrays.insert("LocA(0)", Value::Float(loca_0));
    arrays.insert("HOTALocA(0)", Value::Float(hota_0 * loca_0));
    arrays
}

/// Per-sequence state, before it becomes `Scores` in `eval_sequence`'s dict order.
struct SequenceArrays {
    float: [Alphas; 9],
    tp: Alphas,
    fn_: Alphas,
    fp: Alphas,
}

impl SequenceArrays {
    fn zeros() -> Self {
        Self {
            float: [[0.0; NUM_ALPHAS]; 9],
            tp: [0.0; NUM_ALPHAS],
            fn_: [0.0; NUM_ALPHAS],
            fp: [0.0; NUM_ALPHAS],
        }
    }

    fn float_mut(&mut self, field: &str) -> &mut Alphas {
        let index = FLOAT_ARRAY_FIELDS
            .iter()
            .position(|f| *f == field)
            .expect("a float array field");
        &mut self.float[index]
    }

    /// The dict `eval_sequence` builds: the float arrays, then the counts.
    fn into_scores(self) -> Scores {
        let mut scores = Scores::new();
        for (field, values) in FLOAT_ARRAY_FIELDS.iter().zip(self.float) {
            scores.insert(*field, Value::Array(values.to_vec()));
        }
        scores.insert("HOTA_TP", Value::Array(self.tp.to_vec()));
        scores.insert("HOTA_FN", Value::Array(self.fn_.to_vec()));
        scores.insert("HOTA_FP", Value::Array(self.fp.to_vec()));
        scores
    }
}

/// `a[idx] += values` for a numpy fancy index: every new value is read from the array
/// before any is written, then the writes land in order, so a repeated index keeps its
/// last write instead of accumulating.
fn fancy_add(target: &mut [f64], updates: impl Iterator<Item = (usize, f64)>) {
    let writes: Vec<(usize, f64)> = updates
        .map(|(index, delta)| (index, target[index] + delta))
        .collect();
    for (index, value) in writes {
        target[index] = value;
    }
}

impl Hota {
    /// `eval_sequence`, which also hands each frame's assignment cost matrix
    /// (`-score_mat`, C order) to `on_cost` before it is solved. The tests use it to check
    /// the global alignment score, which reaches the outputs only through the assignment.
    fn eval_sequence_observed(
        &self,
        data: &SequenceData,
        on_cost: &mut dyn FnMut(&[f64]),
    ) -> Result<Scores, MetricError> {
        let mut arrays = SequenceArrays::zeros();

        if data.num_pred_dets() == 0 {
            arrays.fn_ = [data.num_gt_dets() as f64; NUM_ALPHAS];
            *arrays.float_mut("LocA") = [1.0; NUM_ALPHAS];
            return Ok(finalize(arrays.into_scores()));
        }
        if data.num_gt_dets() == 0 {
            arrays.fp = [data.num_pred_dets() as f64; NUM_ALPHAS];
            *arrays.float_mut("LocA") = [1.0; NUM_ALPHAS];
            return Ok(finalize(arrays.into_scores()));
        }

        let num_gt_ids = data.num_gt_ids();
        let num_pred_ids = data.num_pred_ids();
        let mut potential_matches_count = vec![0.0; num_gt_ids * num_pred_ids];
        let mut gt_id_count = vec![0.0; num_gt_ids];
        let mut pred_id_count = vec![0.0; num_pred_ids];

        for ((gt_ids_t, pred_ids_t), similarity) in data
            .gt_ids()
            .iter()
            .zip(data.pred_ids())
            .zip(data.similarity())
        {
            let (n, m) = (gt_ids_t.len(), pred_ids_t.len());
            if n > 0 && m > 0 {
                let sim = similarity.data();
                // similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
                let col_sums = numeric::sum_axis0(sim, n, m);
                let row_sums = numeric::sum_last_axis(sim, m);
                let mut sim_iou = vec![0.0; n * m];
                for (r, &row_sum) in row_sums.iter().enumerate() {
                    for (c, &col_sum) in col_sums.iter().enumerate() {
                        let k = r * m + c;
                        let denom = (col_sum + row_sum) - sim[k];
                        // sim_iou_mask = sim_iou_denom > 0 + EPS
                        if denom > 0.0 + EPS {
                            sim_iou[k] = sim[k] / denom;
                        }
                    }
                }
                // potential_matches_count[gt_ids_t[:, None], pred_ids_t[None, :]] += sim_iou
                fancy_add(
                    &mut potential_matches_count,
                    (0..n * m).map(|k| {
                        let (r, c) = (k / m, k % m);
                        (gt_ids_t[r] * num_pred_ids + pred_ids_t[c], sim_iou[k])
                    }),
                );
            }
            // gt_id_count[gt_ids_t] += 1; pred_id_count[0, pred_ids_t] += 1
            fancy_add(&mut gt_id_count, gt_ids_t.iter().map(|&g| (g, 1.0)));
            fancy_add(&mut pred_id_count, pred_ids_t.iter().map(|&p| (p, 1.0)));
        }

        // potential_matches_count / (gt_id_count + pred_id_count - potential_matches_count)
        let global_alignment_score: Vec<f64> = potential_matches_count
            .iter()
            .enumerate()
            .map(|(k, &pmc)| {
                let (g, p) = (k / num_pred_ids, k % num_pred_ids);
                pmc / ((gt_id_count[g] + pred_id_count[p]) - pmc)
            })
            .collect();

        let alphas = alphas();
        let alpha_thresholds: Alphas = alphas.map(|alpha| alpha - EPS);
        let num_pairs = num_gt_ids * num_pred_ids;
        // np.bincount over alpha_idx * (G * P) + gt_id * P + pred_id
        let mut matches_counts = vec![0u32; NUM_ALPHAS * num_pairs];
        let mut loca = [0.0; NUM_ALPHAS];

        for ((gt_ids_t, pred_ids_t), similarity) in data
            .gt_ids()
            .iter()
            .zip(data.pred_ids())
            .zip(data.similarity())
        {
            let (n, m) = (gt_ids_t.len(), pred_ids_t.len());
            if n == 0 {
                for fp in &mut arrays.fp {
                    *fp += m as f64;
                }
                continue;
            }
            if m == 0 {
                for fn_ in &mut arrays.fn_ {
                    *fn_ += n as f64;
                }
                continue;
            }

            let sim = similarity.data();
            // -(global_alignment_score[gt_ids_t[:, None], pred_ids_t[None, :]] * similarity)
            let mut neg_score = vec![0.0; n * m];
            for r in 0..n {
                for c in 0..m {
                    let k = r * m + c;
                    let gas = global_alignment_score[gt_ids_t[r] * num_pred_ids + pred_ids_t[c]];
                    neg_score[k] = -(gas * sim[k]);
                }
            }
            on_cost(&neg_score);
            let (match_rows, match_cols) = linear_sum_assignment(&neg_score, n, m, false)?;
            let matched_sims: Vec<f64> = match_rows
                .iter()
                .zip(&match_cols)
                .map(|(&r, &c)| sim[r * m + c])
                .collect();

            for a in 0..NUM_ALPHAS {
                // matched = matched_sims >= alpha - EPS
                let mut num_matches = 0usize;
                // np.add.accumulate(np.where(matched, matched_sims, 0.0), axis=0)[-1]: a
                // left-to-right scan that starts from its first element.
                let mut frame_sum = 0.0;
                for (k, &s) in matched_sims.iter().enumerate() {
                    let is_match = s >= alpha_thresholds[a];
                    let term = if is_match { s } else { 0.0 };
                    frame_sum = if k == 0 { term } else { frame_sum + term };
                    if is_match {
                        num_matches += 1;
                        let pair =
                            gt_ids_t[match_rows[k]] * num_pred_ids + pred_ids_t[match_cols[k]];
                        matches_counts[a * num_pairs + pair] += 1;
                    }
                }
                arrays.tp[a] += num_matches as f64;
                arrays.fn_[a] += (n - num_matches) as f64;
                arrays.fp[a] += (m - num_matches) as f64;
                if !matched_sims.is_empty() {
                    loca[a] += frame_sum;
                }
            }
        }

        let mut ass_a = [0.0; NUM_ALPHAS];
        let mut ass_re = [0.0; NUM_ALPHAS];
        let mut ass_pr = [0.0; NUM_ALPHAS];
        let mut products = vec![0.0; num_pairs];
        for a in 0..NUM_ALPHAS {
            let counts = &matches_counts[a * num_pairs..(a + 1) * num_pairs];
            let tp_denominator = maximum(1.0, arrays.tp[a]);
            // np.sum(matches_count * ass_a), ass_a = mc / np.maximum(1, gt + pred - mc)
            for (k, product) in products.iter_mut().enumerate() {
                let mc = f64::from(counts[k]);
                let (g, p) = (k / num_pred_ids, k % num_pred_ids);
                *product = mc * (mc / maximum(1.0, (gt_id_count[g] + pred_id_count[p]) - mc));
            }
            ass_a[a] = numeric::sum(&products) / tp_denominator;
            // np.sum(matches_count * ass_re), ass_re = mc / np.maximum(1, gt_id_count)
            for (k, product) in products.iter_mut().enumerate() {
                let mc = f64::from(counts[k]);
                *product = mc * (mc / maximum(1.0, gt_id_count[k / num_pred_ids]));
            }
            ass_re[a] = numeric::sum(&products) / tp_denominator;
            // np.sum(matches_count * ass_pr), ass_pr = mc / np.maximum(1, pred_id_count)
            for (k, product) in products.iter_mut().enumerate() {
                let mc = f64::from(counts[k]);
                *product = mc * (mc / maximum(1.0, pred_id_count[k % num_pred_ids]));
            }
            ass_pr[a] = numeric::sum(&products) / tp_denominator;
        }
        *arrays.float_mut("AssA") = ass_a;
        *arrays.float_mut("AssRe") = ass_re;
        *arrays.float_mut("AssPr") = ass_pr;
        // np.maximum(1e-10, LocA) / np.maximum(1e-10, HOTA_TP)
        let tp = arrays.tp;
        *arrays.float_mut("LocA") =
            std::array::from_fn(|a| maximum(1e-10, loca[a]) / maximum(1e-10, tp[a]));

        let mut scores = arrays.into_scores();
        compute_det_fields(&mut scores);
        Ok(finalize(scores))
    }
}

impl Metric for Hota {
    fn name(&self) -> &'static str {
        "HOTA"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError> {
        self.eval_sequence_observed(data, &mut |_| {})
    }

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "HOTA" });
        }
        let mut arrays = Scores::new();
        for field in INTEGER_ARRAY_FIELDS {
            arrays.insert(field, combine_sum(all_res, field)?);
        }
        for field in ["AssRe", "AssPr", "AssA"] {
            let value = combine_weighted_av(all_res, field, &arrays, "HOTA_TP")?;
            arrays.insert(field, value);
        }
        // sum(scores["LocA"] * scores["HOTA_TP"] for scores in all_res.values()): builtin
        // sum over arrays, elementwise in order from 0.
        let mut loca_weighted_sum = [0.0; NUM_ALPHAS];
        for res in all_res {
            res.require("LocA")?;
            res.require("HOTA_TP")?;
            let (loca, tp) = (alpha_array(res, "LocA"), alpha_array(res, "HOTA_TP"));
            for a in 0..NUM_ALPHAS {
                loca_weighted_sum[a] += loca[a] * tp[a];
            }
        }
        let tp = alpha_array(&arrays, "HOTA_TP");
        let loca: Vec<f64> = (0..NUM_ALPHAS)
            .map(|a| maximum(1e-10, loca_weighted_sum[a]) / maximum(1e-10, tp[a]))
            .collect();
        arrays.insert("LocA", Value::Array(loca));
        compute_det_fields(&mut arrays);
        Ok(finalize(arrays))
    }

    fn combine_classes_class_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "HOTA" });
        }
        let mut scores = Scores::new();
        for field in INTEGER_ARRAY_FIELDS {
            scores.insert(field, combine_sum(all_res, field)?);
        }
        // np.mean(np.asarray([scores[field] ...]), axis=0) over a (classes, 19) array: row
        // by row, in order.
        for field in FLOAT_ARRAY_FIELDS {
            require_all(all_res, field)?;
            let stacked: Vec<f64> = all_res
                .iter()
                .flat_map(|res| alpha_array(res, field))
                .collect();
            let mean = numeric::mean_axis0(&stacked, all_res.len(), NUM_ALPHAS);
            scores.insert(field, Value::Array(mean));
        }
        // float(np.mean(np.asarray([scores[field] ...]), axis=0)) over a 1-D array: pairwise.
        for field in FLOAT_FIELDS {
            let values = all_res
                .iter()
                .map(|res| res.require(field).map(Value::float))
                .collect::<Result<Vec<f64>, _>>()?;
            scores.insert(field, Value::Float(numeric::mean(&values)));
        }
        Ok(scores)
    }

    fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        self.combine_sequences(all_res)
    }
}

/// Readers and generators for `tests/integration/data/hota_trackmap_internals.json`
/// (`scripts/testdata/dump_metric_internals.py`), shared by the HOTA and TrackMAP unit tests.
#[cfg(test)]
pub(crate) mod internals_testdata {
    use std::path::Path;

    use serde_json::Value as Json;

    use crate::metrics::Scores;

    /// splitmix64, draw for draw the same as `scripts/testdata/testgen.py`.
    pub(crate) struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        pub(crate) fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        pub(crate) fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// An integer in `[0, n)`.
        pub(crate) fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }

        /// A float in `[0, 1)` with 53 random bits (`2**-53` is exact).
        pub(crate) fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
        }
    }

    /// FNV-1a over each word's little-endian bytes.
    pub(crate) fn fnv1a64_words(words: impl IntoIterator<Item = u64>) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for word in words {
            for byte in word.to_le_bytes() {
                h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        h
    }

    pub(crate) fn load() -> Json {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/integration/data/hota_trackmap_internals.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
        let json: Json = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(json["format"], 1);
        json
    }

    pub(crate) fn u64_of(json: &Json) -> u64 {
        json.as_u64().unwrap_or_else(|| panic!("not a u64: {json}"))
    }

    /// `[(field, shape, fnv of the bits)]` in insertion order, as the dump digests scores.
    pub(crate) fn digest(scores: &Scores) -> Vec<(String, Vec<u64>, u64)> {
        scores
            .iter()
            .map(|(field, value)| {
                let shape = value.shape().into_iter().map(|n| n as u64).collect();
                let bits = value.as_slice().iter().map(|x| x.to_bits());
                (field.to_string(), shape, fnv1a64_words(bits))
            })
            .collect()
    }

    pub(crate) fn expected_digest(json: &Json) -> Vec<(String, Vec<u64>, u64)> {
        json.as_array()
            .expect("a digest list")
            .iter()
            .map(|row| {
                let shape = row[1].as_array().expect("shape").iter().map(u64_of);
                (
                    row[0].as_str().expect("field").to_string(),
                    shape.collect(),
                    u64_of(&row[2]),
                )
            })
            .collect()
    }

    /// The digest of class scores as the dump digests its generated inputs: every value's
    /// bits, class by class, field by field in insertion order.
    pub(crate) fn scores_inputs_fnv(all: &[Scores]) -> u64 {
        fnv1a64_words(
            all.iter()
                .flat_map(|scores| scores.iter())
                .flat_map(|(_, value)| value.as_slice().iter().map(|x| x.to_bits())),
        )
    }

    /// Fisher-Yates, from the last position down.
    pub(crate) fn shuffle<T>(rng: &mut SplitMix64, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::internals_testdata::{
        SplitMix64, digest, expected_digest, fnv1a64_words, load, scores_inputs_fnv, shuffle,
        u64_of,
    };
    use super::*;
    use crate::model::{Geometry, Matrix, SequenceDataParts};

    #[test]
    fn fancy_add_keeps_the_last_write_for_a_repeated_index() {
        let mut target = vec![1.0, 10.0];
        fancy_add(&mut target, [(0, 2.0), (0, 5.0), (1, 1.0)].into_iter());
        assert_eq!(target, [6.0, 11.0]);
    }

    /// `frame_ids` in `dump_metric_internals.py`.
    fn frame_ids(rng: &mut SplitMix64, num_ids: usize) -> Vec<usize> {
        let count = if rng.below(5) != 0 {
            8 + rng.below(num_ids as u64 - 7)
        } else {
            rng.below(3)
        };
        let mut ids: Vec<usize> = (0..num_ids).collect();
        shuffle(rng, &mut ids);
        ids.truncate(count as usize);
        if !ids.is_empty() && rng.below(8) == 0 {
            ids.push(ids[0]);
        }
        ids
    }

    /// `hota_dense_sequence` in `dump_metric_internals.py`.
    fn dense_sequence(rng: &mut SplitMix64, name: &str) -> SequenceData {
        let num_timesteps = 4 + rng.below(5) as usize;
        let num_gt_ids = 8 + rng.below(10) as usize;
        let num_pred_ids = 8 + rng.below(17) as usize;
        let (mut gt_ids, mut pred_ids, mut similarity) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..num_timesteps {
            let gt = frame_ids(rng, num_gt_ids);
            let pred = frame_ids(rng, num_pred_ids);
            let sim: Vec<f64> = (0..gt.len() * pred.len())
                .map(|_| {
                    if rng.below(4) == 0 {
                        0.0
                    } else {
                        rng.uniform()
                    }
                })
                .collect();
            similarity.push(Matrix::new(gt.len(), pred.len(), sim).unwrap());
            gt_ids.push(gt);
            pred_ids.push(pred);
        }
        let zeros = |ids: &Vec<Vec<usize>>| -> Vec<Vec<[f64; 4]>> {
            ids.iter().map(|f| vec![[0.0; 4]; f.len()]).collect()
        };
        SequenceData::new(SequenceDataParts {
            name: name.to_string(),
            num_timesteps,
            num_gt_ids,
            num_pred_ids,
            num_gt_dets: gt_ids.iter().map(Vec::len).sum(),
            num_pred_dets: pred_ids.iter().map(Vec::len).sum(),
            pred_confidences: pred_ids.iter().map(|f| vec![1.0; f.len()]).collect(),
            geometry: Geometry::Boxes {
                gt: zeros(&gt_ids),
                pred: zeros(&pred_ids),
            },
            gt_ids,
            pred_ids,
            similarity,
        })
        .unwrap()
    }

    /// `sequence_words` in `dump_metric_internals.py`.
    fn sequence_words(seq: &SequenceData) -> Vec<u64> {
        let mut words = Vec::new();
        for ((gt, pred), sim) in seq
            .gt_ids()
            .iter()
            .zip(seq.pred_ids())
            .zip(seq.similarity())
        {
            words.push(gt.len() as u64);
            words.extend(gt.iter().map(|&id| id as u64));
            words.push(pred.len() as u64);
            words.extend(pred.iter().map(|&id| id as u64));
            words.extend(sim.data().iter().map(|x| x.to_bits()));
        }
        words
    }

    #[test]
    fn dense_sequences_match_python_costs_and_scores() {
        let json = load();
        let cases = json["hota_dense"].as_array().unwrap();
        assert_eq!(cases.len(), 24);
        let mut total_costs = 0;
        for case in cases {
            let seed = u64_of(&case["seed"]);
            let mut rng = SplitMix64::new(seed);
            let sequences: Vec<SequenceData> = (0..3)
                .map(|i| dense_sequence(&mut rng, &format!("dense{i}")))
                .collect();
            assert_eq!(
                fnv1a64_words(sequences.iter().flat_map(sequence_words)),
                u64_of(&case["inputs_fnv"]),
                "seed {seed}: generator drift"
            );
            let mut costs: Vec<u64> = Vec::new();
            let per_sequence: Vec<Scores> = sequences
                .iter()
                .map(|seq| {
                    let mut record = |cost: &[f64]| costs.extend(cost.iter().map(|x| x.to_bits()));
                    Hota.eval_sequence_observed(seq, &mut record).unwrap()
                })
                .collect();
            assert_eq!(
                costs.len() as u64,
                u64_of(&case["num_costs"]),
                "seed {seed}"
            );
            assert_eq!(
                fnv1a64_words(costs.iter().copied()),
                u64_of(&case["costs_fnv"]),
                "seed {seed}: assignment costs differ from Python's"
            );
            total_costs += costs.len();
            for (i, scores) in per_sequence.iter().enumerate() {
                let want = expected_digest(&case["per_sequence"][i]);
                assert_eq!(digest(scores), want, "seed {seed}: sequence {i}");
            }
            let refs: Vec<&Scores> = per_sequence.iter().collect();
            let combined = Hota.combine_sequences(&refs).unwrap();
            assert_eq!(digest(&combined), expected_digest(&case["combined"]));
        }
        assert!(total_costs > 10_000, "only {total_costs} cost entries");
    }

    /// `hota_class_scores` in `dump_metric_internals.py`.
    fn class_scores(rng: &mut SplitMix64) -> Scores {
        let mut scores = Scores::new();
        for field in FLOAT_ARRAY_FIELDS {
            scores.insert(
                field,
                Value::Array((0..19).map(|_| rng.uniform()).collect()),
            );
        }
        for field in INTEGER_ARRAY_FIELDS {
            let counts = (0..19).map(|_| rng.below(500) as f64).collect();
            scores.insert(field, Value::Array(counts));
        }
        for field in FLOAT_FIELDS {
            scores.insert(field, Value::Float(rng.uniform()));
        }
        scores
    }

    #[test]
    fn class_combiners_match_python_with_many_classes() {
        let json = load();
        let cases = json["hota_class_combiners"].as_array().unwrap();
        assert_eq!(cases.len(), 40);
        for case in cases {
            let seed = u64_of(&case["seed"]);
            let mut rng = SplitMix64::new(seed);
            let num_classes = 8 + rng.below(13);
            assert_eq!(num_classes, u64_of(&case["num_classes"]));
            let all: Vec<Scores> = (0..num_classes).map(|_| class_scores(&mut rng)).collect();
            let drift = "generator drift";
            assert_eq!(
                scores_inputs_fnv(&all),
                u64_of(&case["inputs_fnv"]),
                "{seed}: {drift}"
            );
            let refs: Vec<&Scores> = all.iter().collect();
            let class_averaged = Hota.combine_classes_class_averaged(&refs).unwrap();
            let det_averaged = Hota.combine_classes_det_averaged(&refs).unwrap();
            assert_eq!(
                digest(&class_averaged),
                expected_digest(&case["class_averaged"]),
                "seed {seed}: class_averaged"
            );
            assert_eq!(
                digest(&det_averaged),
                expected_digest(&case["det_averaged"]),
                "seed {seed}: det_averaged"
            );
        }
    }
}
