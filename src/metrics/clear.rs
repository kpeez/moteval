//! CLEAR (MOTChallenge) metric family: MOTA, MOTP, IDSW, MT/PT/ML, and friends, ported
//! from `moteval/metrics/clear.py`.
//!
//! Each frame is matched with scipy's Hungarian solver on a score matrix that gives a
//! `1000x` bonus to the GT-track/pred-ID pairing of the previous timestep, after the
//! eps-guarded threshold (`similarity < threshold - eps` scores `0`). Every reduction
//! copies the order of its Python call site.

use crate::lsap::linear_sum_assignment;
use crate::metrics::{Metric, MetricError, Scores, Value, combine_sum};
use crate::model::SequenceData;
use crate::numeric::{EPS, log10, mean, plain_sum, py_max, sum};

/// The public fields, in the Python `fields` order (`_FLOAT_FIELDS + _INTEGER_FIELDS`).
pub const FIELDS: &[&str] = &[
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "sMOTA",
    "CLR_F1",
    "FP_per_frame",
    "MOTAL",
    "MOTP_sum",
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
    "CLR_Frames",
];

/// `THRESHOLD`: the similarity a match needs, guarded by `EPS`.
pub const THRESHOLD: f64 = 0.5;

/// `_INTEGER_FIELDS`: counts, summed by every combiner.
const INTEGER_FIELDS: &[&str] = &[
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
    "CLR_Frames",
];

/// `_FLOAT_FIELDS`: averaged by `combine_classes_class_averaged`.
const FLOAT_FIELDS: &[&str] = &[
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "sMOTA",
    "CLR_F1",
    "FP_per_frame",
    "MOTAL",
    "MOTP_sum",
];

/// `_SUMMED_FIELDS`: what `combine_sequences` sums before it recomputes the rest.
const SUMMED_FIELDS: &[&str] = &[
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
    "CLR_Frames",
    "MOTP_sum",
];

/// `moteval.metrics.clear.CLEAR`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Clear;

/// `numpy` fancy-index `counts[ids] += 1`: each distinct id gains one, however often it
/// repeats in `ids`. `stamp` remembers the frame that last touched each id.
fn add_once_per_frame(counts: &mut [f64], stamp: &mut [usize], ids: &[usize], frame: usize) {
    for &id in ids {
        if stamp[id] != frame {
            stamp[id] = frame;
            counts[id] += 1.0;
        }
    }
}

/// `CLEAR._compute_final_fields`: derive the ratios from the counts, inserting them in
/// the Python order. Python's `max(1.0, x)` is `py_max`.
fn compute_final_fields(res: &mut Scores) {
    let mt = res.float("MT");
    let ml = res.float("ML");
    let pt = res.float("PT");
    let tp = res.float("CLR_TP");
    let fn_ = res.float("CLR_FN");
    let fp = res.float("CLR_FP");
    let idsw = res.float("IDSW");
    let motp_sum = res.float("MOTP_sum");
    let frames = res.float("CLR_Frames");

    let num_gt_ids = mt + ml + pt;
    let set = |res: &mut Scores, field: &str, x: f64| res.insert(field, Value::Float(x));
    set(res, "MTR", mt / py_max(1.0, num_gt_ids));
    set(res, "MLR", ml / py_max(1.0, num_gt_ids));
    set(res, "PTR", pt / py_max(1.0, num_gt_ids));
    set(res, "CLR_Re", tp / py_max(1.0, tp + fn_));
    set(res, "CLR_Pr", tp / py_max(1.0, tp + fp));
    set(res, "MODA", (tp - fp) / py_max(1.0, tp + fn_));
    set(res, "MOTA", (tp - fp - idsw) / py_max(1.0, tp + fn_));
    set(res, "MOTP", motp_sum / py_max(1.0, tp));
    set(res, "sMOTA", (motp_sum - fp - idsw) / py_max(1.0, tp + fn_));
    set(res, "CLR_F1", tp / py_max(1.0, tp + 0.5 * fn_ + 0.5 * fp));
    set(res, "FP_per_frame", fp / py_max(1.0, frames));
    // np.log10 differs from libm log10 on some hosts; see numeric::log10.
    let safe_log_idsw = if idsw > 0.0 { log10(idsw) } else { idsw };
    set(
        res,
        "MOTAL",
        (tp - fp - safe_log_idsw) / py_max(1.0, tp + fn_),
    );
}

impl Metric for Clear {
    fn name(&self) -> &'static str {
        "CLEAR"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError> {
        // dict.fromkeys(self.fields, 0.0)
        let mut res = Scores::new();
        for &field in FIELDS {
            res.insert(field, Value::Float(0.0));
        }

        if data.num_pred_dets() == 0 {
            res.insert("CLR_FN", Value::Float(data.num_gt_dets() as f64));
            res.insert("ML", Value::Float(data.num_gt_ids() as f64));
            res.insert("MLR", Value::Float(1.0));
            return Ok(res);
        }
        if data.num_gt_dets() == 0 {
            res.insert("CLR_FP", Value::Float(data.num_pred_dets() as f64));
            res.insert("MLR", Value::Float(1.0));
            return Ok(res);
        }

        let num_gt_ids = data.num_gt_ids();
        let mut gt_id_count = vec![0.0; num_gt_ids];
        let mut gt_matched_count = vec![0.0; num_gt_ids];
        let mut gt_frag_count = vec![0.0; num_gt_ids];
        let mut id_count_stamp = vec![usize::MAX; num_gt_ids];
        let mut matched_stamp = vec![usize::MAX; num_gt_ids];
        let mut frag_stamp = vec![usize::MAX; num_gt_ids];

        // IDSW is scored against the last frame each gt id was matched (any number of
        // frames back), but only the immediately preceding timestep feeds the bonus.
        // `None` is Python's NaN.
        let mut prev_tracker_id: Vec<Option<usize>> = vec![None; num_gt_ids];
        let mut prev_timestep_tracker_id: Vec<Option<usize>> = vec![None; num_gt_ids];
        // The gt ids whose `prev_timestep_tracker_id` is set, so the per-frame reset to
        // NaN touches only them.
        let mut tracked_now: Vec<usize> = Vec::new();

        let mut clr_tp = 0.0;
        let mut clr_fn = 0.0;
        let mut clr_fp = 0.0;
        let mut idsw = 0.0;
        let mut motp_sum = 0.0;

        let threshold = THRESHOLD - EPS;
        for (t, ((gt_ids_t, pred_ids_t), similarity)) in data
            .gt_ids()
            .iter()
            .zip(data.pred_ids())
            .zip(data.similarity())
            .enumerate()
        {
            if gt_ids_t.is_empty() {
                clr_fp += pred_ids_t.len() as f64;
                continue;
            }
            if pred_ids_t.is_empty() {
                clr_fn += gt_ids_t.len() as f64;
                add_once_per_frame(&mut gt_id_count, &mut id_count_stamp, gt_ids_t, t);
                continue;
            }

            // score_mat = np.where(similarity < THRESHOLD - EPS, 0.0,
            //                      1000 * continues_prev_pairing + similarity)
            let (num_rows, num_cols) = similarity.shape();
            let mut score_mat = Vec::with_capacity(num_rows * num_cols);
            for (i, &gt_id) in gt_ids_t.iter().enumerate() {
                let prev = prev_timestep_tracker_id[gt_id];
                for (&pred_id, &sim) in pred_ids_t.iter().zip(similarity.row(i)) {
                    let score = if sim < threshold {
                        0.0
                    } else {
                        let bonus = if prev == Some(pred_id) { 1000.0 } else { 0.0 };
                        bonus + sim
                    };
                    score_mat.push(score);
                }
            }
            // linear_sum_assignment(-score_mat)
            let negated: Vec<f64> = score_mat.iter().map(|&s| -s).collect();
            let (rows, cols) = linear_sum_assignment(&negated, num_rows, num_cols, false)?;
            // actually_matched_mask = score_mat[match_rows, match_cols] > 0 + EPS
            let matches: Vec<(usize, usize)> = rows
                .into_iter()
                .zip(cols)
                .filter(|&(r, c)| score_mat[r * num_cols + c] > EPS)
                .collect();

            // is_idsw reads prev_tracker_id before this frame writes it.
            let num_idsw = matches
                .iter()
                .filter(|&&(r, c)| {
                    prev_tracker_id[gt_ids_t[r]].is_some_and(|prev| prev != pred_ids_t[c])
                })
                .count();
            idsw += num_idsw as f64;

            add_once_per_frame(&mut gt_id_count, &mut id_count_stamp, gt_ids_t, t);
            let matched_gt_ids: Vec<usize> = matches.iter().map(|&(r, _)| gt_ids_t[r]).collect();
            add_once_per_frame(
                &mut gt_matched_count,
                &mut matched_stamp,
                &matched_gt_ids,
                t,
            );

            // not_previously_tracked, read before the reset; only a gt id matched in this
            // frame can be currently tracked, so only those can gain a fragment.
            let newly_tracked: Vec<usize> = matched_gt_ids
                .iter()
                .copied()
                .filter(|&g| prev_timestep_tracker_id[g].is_none())
                .collect();
            // prev_tracker_id[matched_gt_ids] = matched_pred_ids (the last write wins)
            for &(r, c) in &matches {
                prev_tracker_id[gt_ids_t[r]] = Some(pred_ids_t[c]);
            }
            // prev_timestep_tracker_id[:] = np.nan, then the same assignment
            for g in tracked_now.drain(..) {
                prev_timestep_tracker_id[g] = None;
            }
            for &(r, c) in &matches {
                prev_timestep_tracker_id[gt_ids_t[r]] = Some(pred_ids_t[c]);
                tracked_now.push(gt_ids_t[r]);
            }
            // gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)
            add_once_per_frame(&mut gt_frag_count, &mut frag_stamp, &newly_tracked, t);

            let num_matches = matches.len();
            clr_tp += num_matches as f64;
            clr_fn += (gt_ids_t.len() - num_matches) as f64;
            clr_fp += (pred_ids_t.len() - num_matches) as f64;
            if num_matches > 0 {
                // builtin sum over the np.float64 elements: in order, from 0
                motp_sum += plain_sum(matches.iter().map(|&(r, c)| similarity.get(r, c)));
            }
        }

        // tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        let tracked_ratio: Vec<f64> = gt_matched_count
            .iter()
            .zip(&gt_id_count)
            .filter(|&(_, &count)| count > 0.0)
            .map(|(&matched, &count)| matched / count)
            .collect();
        let mt = tracked_ratio.iter().filter(|&&r| r > 0.8).count() as f64;
        let pt = tracked_ratio.iter().filter(|&&r| r >= 0.2).count() as f64 - mt;
        let ml = num_gt_ids as f64 - mt - pt;
        let frag_minus_one: Vec<f64> = gt_frag_count
            .iter()
            .filter(|&&f| f > 0.0)
            .map(|&f| f - 1.0)
            .collect();

        res.insert("CLR_TP", Value::Float(clr_tp));
        res.insert("CLR_FN", Value::Float(clr_fn));
        res.insert("CLR_FP", Value::Float(clr_fp));
        res.insert("IDSW", Value::Float(idsw));
        res.insert("MOTP_sum", Value::Float(motp_sum));
        res.insert("MT", Value::Float(mt));
        res.insert("PT", Value::Float(pt));
        res.insert("ML", Value::Float(ml));
        res.insert("Frag", Value::Float(sum(&frag_minus_one)));
        res.insert("CLR_Frames", Value::Float(data.num_timesteps() as f64));

        compute_final_fields(&mut res);
        Ok(res)
    }

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "CLEAR" });
        }
        let mut res = Scores::new();
        for &field in SUMMED_FIELDS {
            res.insert(field, combine_sum(all_res, field)?);
        }
        compute_final_fields(&mut res);
        Ok(res)
    }

    fn combine_classes_class_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "CLEAR" });
        }
        // Float fields (MOTP_sum included) are averaged, not recomputed from the counts.
        let mut res = Scores::new();
        for &field in INTEGER_FIELDS {
            res.insert(field, combine_sum(all_res, field)?);
        }
        for &field in FLOAT_FIELDS {
            // float(np.mean(np.asarray([...])))
            let values = all_res
                .iter()
                .map(|r| r.require(field).map(Value::float))
                .collect::<Result<Vec<f64>, _>>()?;
            res.insert(field, Value::Float(mean(&values)));
        }
        Ok(res)
    }

    fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        self.combine_sequences(all_res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_are_float_fields_then_integer_fields() {
        let joined: Vec<&str> = FLOAT_FIELDS.iter().chain(INTEGER_FIELDS).copied().collect();
        assert_eq!(FIELDS, joined.as_slice());
        let summed: Vec<&str> = INTEGER_FIELDS
            .iter()
            .chain(&["MOTP_sum"])
            .copied()
            .collect();
        assert_eq!(SUMMED_FIELDS, summed.as_slice());
    }

    #[test]
    fn fancy_index_increment_counts_each_id_once_per_frame() {
        let mut counts = vec![0.0; 3];
        let mut stamp = vec![usize::MAX; 3];
        add_once_per_frame(&mut counts, &mut stamp, &[0, 2, 0, 0], 0);
        add_once_per_frame(&mut counts, &mut stamp, &[2, 2], 1);
        assert_eq!(counts, [1.0, 0.0, 2.0]);
    }

    /// MOTAL takes `np.log10(IDSW)`. When `TP - FP` is small, the last bit of the
    /// logarithm reaches MOTAL. The expected bits are the `py-reference` CLEAR's for these
    /// counts on an AVX-512 Linux host, where numpy computes `log10` with SVML.
    #[test]
    fn motal_takes_numpy_log10() {
        let cases = [
            (12.0, 12.0, 11.0, 0xbfb6_3764_806b_b1e9_u64),
            (41.0, 38.0, 40.0, 0x3fa1_750b_31b5_4707),
        ];
        for (tp, fp, idsw, svml_bits) in cases {
            let mut res = Scores::new();
            for &field in SUMMED_FIELDS {
                res.insert(field, Value::Float(0.0));
            }
            res.insert("CLR_TP", Value::Float(tp));
            res.insert("CLR_FP", Value::Float(fp));
            res.insert("IDSW", Value::Float(idsw));
            compute_final_fields(&mut res);
            let with_svml = (tp - fp - crate::numeric::log10_svml(idsw)) / tp;
            assert_eq!(with_svml.to_bits(), svml_bits, "SVML kernel, IDSW={idsw}");
            let with_libm = (tp - fp - idsw.log10()) / tp;
            let want = if crate::numeric::numpy_uses_svml() {
                // glibc's log10 gives other bits here, so the case tells the two apart.
                assert_ne!(with_libm.to_bits(), svml_bits, "glibc, IDSW={idsw}");
                svml_bits
            } else {
                with_libm.to_bits()
            };
            assert_eq!(res.float("MOTAL").to_bits(), want, "MOTAL for IDSW={idsw}");
        }
    }

    #[test]
    fn combiners_reject_no_results() {
        assert!(matches!(
            Clear.combine_sequences(&[]),
            Err(MetricError::NoResults { metric: "CLEAR" })
        ));
        assert!(Clear.combine_classes_class_averaged(&[]).is_err());
    }
}
