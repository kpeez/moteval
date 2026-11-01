//! Identity metric family: IDF1, IDR, IDP, IDTP, IDFN, IDFP, ported from
//! `moteval/metrics/identity.py`.
//!
//! One global track-level matching per sequence, on the
//! `(num_gt_ids + num_pred_ids) x (num_gt_ids + num_pred_ids)` block cost matrix whose
//! off-diagonal blocks hold `1e10`, solved with scipy's Hungarian solver. The similarity
//! threshold is not eps-guarded (`similarity >= threshold`), as upstream.

use crate::lsap::linear_sum_assignment;
use crate::metrics::{Metric, MetricError, Scores, Value, combine_sum};
use crate::model::SequenceData;
use crate::numeric::{mean, py_max, sum};

/// The public fields, in the Python `fields` order (`_FLOAT_FIELDS + _INTEGER_FIELDS`).
pub const FIELDS: &[&str] = &["IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"];

/// `THRESHOLD`: the similarity a potential match needs (no eps guard).
pub const THRESHOLD: f64 = 0.5;

/// `_INTEGER_FIELDS`: counts, summed by every combiner.
const INTEGER_FIELDS: &[&str] = &["IDTP", "IDFN", "IDFP"];

/// `_FLOAT_FIELDS`: averaged by `combine_classes_class_averaged`.
const FLOAT_FIELDS: &[&str] = &["IDF1", "IDR", "IDP"];

/// `moteval.metrics.identity.Identity`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

/// The `fn_mat` and `fp_mat` of `Identity.eval_sequence`, both `n x n` with
/// `n = num_gt_ids + num_pred_ids`, in C order.
///
/// Python fills them in this order, and each entry ends with the value of its last write:
/// zeros; `fp[G:, :T] = 1e10`; `fn[:G, T:] = 1e10`; per gt id `g`,
/// `fn[g, :T] = gt_count[g]` and `fn[g, T + g] = gt_count[g]`; per pred id `p`,
/// `fp[:G, p] = pred_count[p]` and `fp[p + G, p] = pred_count[p]`; then
/// `fn[:G, :T] -= potential` and `fp[:G, :T] -= potential`.
fn block_matrices(
    gt_id_count: &[f64],
    pred_id_count: &[f64],
    potential_matches_count: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let num_gt_ids = gt_id_count.len();
    let num_pred_ids = pred_id_count.len();
    let n = num_gt_ids + num_pred_ids;
    let mut fp_mat = vec![0.0; n * n];
    let mut fn_mat = vec![0.0; n * n];
    for r in num_gt_ids..n {
        fp_mat[r * n..r * n + num_pred_ids].fill(1e10);
    }
    for r in 0..num_gt_ids {
        fn_mat[r * n + num_pred_ids..(r + 1) * n].fill(1e10);
    }
    for (gt_id, &count) in gt_id_count.iter().enumerate() {
        fn_mat[gt_id * n..gt_id * n + num_pred_ids].fill(count);
        fn_mat[gt_id * n + num_pred_ids + gt_id] = count;
    }
    for (pred_id, &count) in pred_id_count.iter().enumerate() {
        for r in 0..num_gt_ids {
            fp_mat[r * n + pred_id] = count;
        }
        fp_mat[(pred_id + num_gt_ids) * n + pred_id] = count;
    }
    for r in 0..num_gt_ids {
        for c in 0..num_pred_ids {
            let potential = potential_matches_count[r * num_pred_ids + c];
            fn_mat[r * n + c] -= potential;
            fp_mat[r * n + c] -= potential;
        }
    }
    (fn_mat, fp_mat)
}

/// `Identity._compute_final_fields`: derive the ratios, inserting them in the Python
/// order. Python's `max(1.0, x)` is `py_max`.
fn compute_final_fields(res: &mut Scores) {
    let idtp = res.float("IDTP");
    let idfn = res.float("IDFN");
    let idfp = res.float("IDFP");
    res.insert("IDR", Value::Float(idtp / py_max(1.0, idtp + idfn)));
    res.insert("IDP", Value::Float(idtp / py_max(1.0, idtp + idfp)));
    res.insert(
        "IDF1",
        Value::Float(idtp / py_max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)),
    );
}

impl Metric for Identity {
    fn name(&self) -> &'static str {
        "Identity"
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
            res.insert("IDFN", Value::Float(data.num_gt_dets() as f64));
            return Ok(res);
        }
        if data.num_gt_dets() == 0 {
            res.insert("IDFP", Value::Float(data.num_pred_dets() as f64));
            return Ok(res);
        }

        let num_gt_ids = data.num_gt_ids();
        let num_pred_ids = data.num_pred_ids();

        // np.bincount over every (gt, pred) pair with similarity >= THRESHOLD, every
        // frame: each occurrence counts, repeated ids included.
        let mut potential_matches_count = vec![0.0; num_gt_ids * num_pred_ids];
        let mut gt_id_count = vec![0.0; num_gt_ids];
        let mut pred_id_count = vec![0.0; num_pred_ids];
        for ((gt_ids_t, pred_ids_t), similarity) in data
            .gt_ids()
            .iter()
            .zip(data.pred_ids())
            .zip(data.similarity())
        {
            for (i, &gt_id) in gt_ids_t.iter().enumerate() {
                for (&pred_id, &sim) in pred_ids_t.iter().zip(similarity.row(i)) {
                    if sim >= THRESHOLD {
                        potential_matches_count[gt_id * num_pred_ids + pred_id] += 1.0;
                    }
                }
            }
            for &gt_id in gt_ids_t {
                gt_id_count[gt_id] += 1.0;
            }
            for &pred_id in pred_ids_t {
                pred_id_count[pred_id] += 1.0;
            }
        }

        let (fn_mat, fp_mat) =
            block_matrices(&gt_id_count, &pred_id_count, &potential_matches_count);
        let n = num_gt_ids + num_pred_ids;
        let cost: Vec<f64> = fn_mat.iter().zip(&fp_mat).map(|(&a, &b)| a + b).collect();
        let (rows, cols) = linear_sum_assignment(&cost, n, n, false)?;

        // float(fn_mat[match_rows, match_cols].sum()): numpy's pairwise sum
        let matched = |mat: &[f64]| -> Vec<f64> {
            rows.iter()
                .zip(&cols)
                .map(|(&r, &c)| mat[r * n + c])
                .collect()
        };
        let idfn = sum(&matched(&fn_mat));
        let idfp = sum(&matched(&fp_mat));
        res.insert("IDFN", Value::Float(idfn));
        res.insert("IDFP", Value::Float(idfp));
        res.insert("IDTP", Value::Float(sum(&gt_id_count) - idfn));

        compute_final_fields(&mut res);
        Ok(res)
    }

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "Identity" });
        }
        let mut res = Scores::new();
        for &field in INTEGER_FIELDS {
            res.insert(field, combine_sum(all_res, field)?);
        }
        compute_final_fields(&mut res);
        Ok(res)
    }

    fn combine_classes_class_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "Identity" });
        }
        // Float fields are averaged, not recomputed from the summed counts.
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
    }

    /// The block matrices for two gt ids and one pred id, written out by hand from the
    /// Python fill order.
    #[test]
    fn block_matrices_follow_the_python_fill_order() {
        let (fn_mat, fp_mat) = block_matrices(&[3.0, 5.0], &[4.0], &[2.0, 0.0]);
        let big = 1e10;
        #[rustfmt::skip]
        let want_fn = [
            1.0, 3.0, big,
            5.0, big, 5.0,
            0.0, 0.0, 0.0,
        ];
        #[rustfmt::skip]
        let want_fp = [
            2.0, 0.0, 0.0,
            4.0, 0.0, 0.0,
            4.0, 0.0, 0.0,
        ];
        assert_eq!(fn_mat, want_fn);
        assert_eq!(fp_mat, want_fp);
    }

    #[test]
    fn combiners_reject_no_results() {
        assert!(Identity.combine_sequences(&[]).is_err());
        assert!(Identity.combine_classes_class_averaged(&[]).is_err());
    }
}
