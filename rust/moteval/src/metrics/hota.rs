//! HOTA metric family over 19 alpha thresholds, plus HOTA(0), LocA(0), HOTALocA(0),
//! ported from `moteval/metrics/hota.py` with all three combiners.
//!
//! TrackEval's global alignment (a Hungarian assignment weighted by a Jaccard-normalized
//! potential-match count) runs once per frame; the eps-guarded `similarity >= alpha - eps`
//! test then scores the 19 alphas. Every sum follows the reduction order of its Python
//! line (see the comments that quote it).

use crate::lsap::linear_sum_assignment;
use crate::metrics::{Metric, MetricError, Scores, Value, combine_sum, combine_weighted_av};
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

impl Metric for Hota {
    fn name(&self) -> &'static str {
        "HOTA"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError> {
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

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "HOTA" });
        }
        let mut arrays = Scores::new();
        for field in INTEGER_ARRAY_FIELDS {
            arrays.insert(field, combine_sum(all_res, field));
        }
        for field in ["AssRe", "AssPr", "AssA"] {
            let value = combine_weighted_av(all_res, field, &arrays, "HOTA_TP");
            arrays.insert(field, value);
        }
        // sum(scores["LocA"] * scores["HOTA_TP"] for scores in all_res.values()): builtin
        // sum over arrays, elementwise in order from 0.
        let mut loca_weighted_sum = [0.0; NUM_ALPHAS];
        for res in all_res {
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
            scores.insert(field, combine_sum(all_res, field));
        }
        // np.mean(np.asarray([scores[field] ...]), axis=0) over a (classes, 19) array: row
        // by row, in order.
        for field in FLOAT_ARRAY_FIELDS {
            let stacked: Vec<f64> = all_res
                .iter()
                .flat_map(|res| alpha_array(res, field))
                .collect();
            let mean = numeric::mean_axis0(&stacked, all_res.len(), NUM_ALPHAS);
            scores.insert(field, Value::Array(mean));
        }
        // float(np.mean(np.asarray([scores[field] ...]), axis=0)) over a 1-D array: pairwise.
        for field in FLOAT_FIELDS {
            let values: Vec<f64> = all_res.iter().map(|res| res.float(field)).collect();
            scores.insert(field, Value::Float(numeric::mean(&values)));
        }
        Ok(scores)
    }

    fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        self.combine_sequences(all_res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fancy_add_keeps_the_last_write_for_a_repeated_index() {
        let mut target = vec![1.0, 10.0];
        fancy_add(&mut target, [(0, 2.0), (0, 5.0), (1, 1.0)].into_iter());
        assert_eq!(target, [6.0, 11.0]);
    }
}
