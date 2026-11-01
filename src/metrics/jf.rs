use crate::lsap::linear_sum_assignment;
use crate::metrics::{Metric, MetricError, Scores, Value, combine_sum, combine_weighted_av};
use crate::model::{GeometryKind, SequenceData};
use crate::numeric::{self, EPS};
use crate::rle::{self, RleMask};

pub const FIELDS: &[&str] = &[
    "J-Mean",
    "J-Recall",
    "J-Decay",
    "F-Mean",
    "F-Recall",
    "F-Decay",
    "J&F",
    "num_gt_tracks",
];

const FLOAT_FIELDS: [&str; 7] = [
    "J-Mean", "J-Recall", "J-Decay", "F-Mean", "F-Recall", "F-Decay", "J&F",
];

const BOUND_TH: f64 = 0.008;

const N_BINS: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct JAndF;

impl Metric for JAndF {
    fn name(&self) -> &'static str {
        "JAndF"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn geometry(&self) -> Option<GeometryKind> {
        Some(GeometryKind::Masks)
    }

    fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError> {
        eval_sequence(data)
    }

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "JAndF" });
        }
        let mut res = Scores::new();
        res.insert("num_gt_tracks", combine_sum(all_res, "num_gt_tracks")?);
        for field in FLOAT_FIELDS {
            let value = combine_weighted_av(all_res, field, &res, "num_gt_tracks")?;
            res.insert(field, value);
        }
        Ok(res)
    }

    fn combine_classes_class_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "JAndF" });
        }
        let mut res = Scores::new();
        res.insert("num_gt_tracks", combine_sum(all_res, "num_gt_tracks")?);
        for field in FLOAT_FIELDS {
            let values = all_res
                .iter()
                .map(|scores| scores.require(field).map(Value::float))
                .collect::<Result<Vec<f64>, _>>()?;
            res.insert(field, Value::Float(numeric::mean(&values)));
        }
        Ok(res)
    }

    fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        // TrackEval's J&F det-averaged combiner is the class mean too.
        self.combine_classes_class_averaged(all_res)
    }
}

fn eval_sequence(data: &SequenceData) -> Result<Scores, MetricError> {
    let num_timesteps = data.num_timesteps();
    let num_gt_ids = data.num_gt_ids();
    let mut num_pred_ids = data.num_pred_ids();
    let gt_masks = data.gt_masks()?;
    let pred_masks = data.pred_masks()?;

    let first_size = |ids: &[Vec<usize>], masks: &[Vec<RleMask>]| {
        (0..num_timesteps)
            .find(|&t| !ids[t].is_empty())
            .map(|t| (masks[t][0].height(), masks[t][0].width()))
    };
    let frame_shape = if num_gt_ids > 0 {
        first_size(data.gt_ids(), gt_masks)
    } else if num_pred_ids > 0 {
        first_size(data.pred_ids(), pred_masks)
    } else {
        None
    };

    let padding = frame_shape.map(|(h, w)| rle::encode(&vec![0u8; h * w], h, w));
    let mut gt_dets: Vec<Vec<&RleMask>> = gt_masks.iter().map(|f| f.iter().collect()).collect();
    let mut pred_dets: Vec<Vec<&RleMask>> = pred_masks.iter().map(|f| f.iter().collect()).collect();
    if let Some(padding) = &padding {
        for t in 0..num_timesteps {
            gt_dets[t] = by_id(&data.gt_ids()[t], &gt_masks[t], num_gt_ids, padding);
            pred_dets[t] = by_id(&data.pred_ids()[t], &pred_masks[t], num_pred_ids, padding);
        }
        if num_pred_ids < num_gt_ids {
            let diff = num_gt_ids - num_pred_ids;
            for frame in &mut pred_dets {
                frame.extend(std::iter::repeat_n(padding, diff));
            }
            num_pred_ids += diff;
        }
    }

    let j = compute_j(
        &gt_dets,
        &pred_dets,
        num_gt_ids,
        num_pred_ids,
        num_timesteps,
    )?;
    let gt_areas: Vec<Vec<u32>> = gt_dets.iter().map(|f| areas(f)).collect();
    let pred_areas: Vec<Vec<u32>> = pred_dets.iter().map(|f| areas(f)).collect();

    let optim = if num_timesteps == 0 {
        vec![f64::NAN; num_pred_ids * num_gt_ids]
    } else {
        numeric::mean_last_axis(&j, num_timesteps)
    };
    let neg: Vec<f64> = optim.iter().map(|&x| -x).collect();
    let (row_ind, col_ind) = linear_sum_assignment(&neg, num_pred_ids, num_gt_ids, false)?;

    let mut j_m: Vec<Vec<f64>> = Vec::with_capacity(num_gt_ids.max(row_ind.len()));
    let mut f_m: Vec<Vec<f64>> = Vec::with_capacity(num_gt_ids.max(row_ind.len()));
    for (&pred_ind, &gt_ind) in row_ind.iter().zip(&col_ind) {
        let start = (pred_ind * num_gt_ids + gt_ind) * num_timesteps;
        j_m.push(j[start..start + num_timesteps].to_vec());
        f_m.push(compute_f(
            &gt_dets,
            &pred_dets,
            pred_ind,
            gt_ind,
            &gt_areas,
            &pred_areas,
        )?);
    }
    while j_m.len() < data.num_gt_ids() {
        j_m.push(vec![0.0; num_timesteps]);
        f_m.push(vec![0.0; num_timesteps]);
    }

    // TrackEval casts the decay-bin edges to `uint8`, so they wrap past 255 frames.
    let ids: Vec<u8> = numeric::linspace(1.0, num_timesteps as f64, N_BINS + 1, true)
        .into_iter()
        .map(|x| numpy_f64_to_u8(numeric::round_half_even(x + 1e-10) - 1.0))
        .collect();
    let bin = |row: &[f64], i: usize| {
        // numpy `uint8` `ids[i + 1] + 1` wraps, and the slice clamps as numpy slicing does.
        let start = usize::from(ids[i]).min(row.len());
        let stop = usize::from(ids[i + 1].wrapping_add(1)).min(row.len());
        numeric::nanmean(&row[start..stop.max(start)])
    };
    let decay = |rows: &[Vec<f64>]| -> Vec<f64> {
        rows.iter()
            .map(|row| bin(row, 0) - bin(row, N_BINS - 1))
            .collect()
    };
    let means =
        |rows: &[Vec<f64>]| -> Vec<f64> { rows.iter().map(|r| numeric::nanmean(r)).collect() };
    let recalls = |rows: &[Vec<f64>]| -> Vec<f64> {
        // numpy `nanmean` on TrackEval's bool array is `mean`.
        rows.iter()
            .map(|row| {
                let hits: Vec<f64> = row
                    .iter()
                    .map(|&x| if x > 0.5 + EPS { 1.0 } else { 0.0 })
                    .collect();
                numeric::mean(&hits)
            })
            .collect()
    };

    let mut scores = Scores::new();
    scores.insert("num_gt_tracks", Value::Float(j_m.len() as f64));
    let per_track = [
        ("J-Mean", means(&j_m)),
        ("J-Recall", recalls(&j_m)),
        ("J-Decay", decay(&j_m)),
        ("F-Mean", means(&f_m)),
        ("F-Recall", recalls(&f_m)),
        ("F-Decay", decay(&f_m)),
    ];
    for (field, values) in per_track {
        scores.insert(field, Value::Float(numeric::mean(&values)));
    }
    let jf = (scores.float("J-Mean") + scores.float("F-Mean")) / 2.0;
    scores.insert("J&F", Value::Float(jf));
    Ok(scores)
}

fn by_id<'a>(
    ids: &[usize],
    dets: &'a [RleMask],
    num_ids: usize,
    padding: &'a RleMask,
) -> Vec<&'a RleMask> {
    let mut out = vec![padding; num_ids];
    for (&id, det) in ids.iter().zip(dets) {
        out[id] = det;
    }
    out
}

fn areas(dets: &[&RleMask]) -> Vec<u32> {
    dets.iter().map(|m| rle::area_u32(m)).collect()
}

/// numpy `astype(np.uint8)` on x86-64: via `int32`, so it wraps mod 256; out of range gives 0.
fn numpy_f64_to_u8(x: f64) -> u8 {
    if (-2_147_483_648.0..2_147_483_648.0).contains(&x) {
        (x as i32) as u8
    } else {
        0
    }
}

/// Empty frames reproduce TrackEval's `_compute_j` numpy errors on a `(1, 0)` IoU array.
fn compute_j(
    gt_dets: &[Vec<&RleMask>],
    pred_dets: &[Vec<&RleMask>],
    num_gt_ids: usize,
    num_pred_ids: usize,
    num_timesteps: usize,
) -> Result<Vec<f64>, MetricError> {
    let mut j = vec![0.0; num_pred_ids * num_gt_ids * num_timesteps];
    for t in 0..num_timesteps {
        let (time_gt, time_pred) = (&gt_dets[t], &pred_dets[t]);
        let (num_gt, num_pred) = (time_gt.len(), time_pred.len());
        if num_gt == 0 || num_pred == 0 {
            if num_pred > 1 {
                return Err(bool_index_error(0, 1, num_pred));
            }
            if num_gt > 0 {
                return Err(bool_index_error(1, 0, num_gt));
            }
            if num_gt_ids != 0 {
                return Err(MetricError::Python {
                    exception: "ValueError",
                    message: format!(
                        "could not broadcast input array from shape (1,0) into shape \
                         ({num_pred_ids},{num_gt_ids})"
                    ),
                });
            }
            continue;
        }
        let area_gt = areas(time_gt);
        let area_pred = areas(time_pred);
        let ious = rle::iou(time_pred, time_gt, &vec![false; num_gt])?;
        if (num_pred, num_gt) != (num_pred_ids, num_gt_ids) {
            return Err(MetricError::Python {
                exception: "ValueError",
                message: format!(
                    "could not broadcast input array from shape ({num_pred},{num_gt}) into \
                     shape ({num_pred_ids},{num_gt_ids})"
                ),
            });
        }
        for p in 0..num_pred {
            for g in 0..num_gt {
                let both_empty = area_pred[p] == 0 && area_gt[g] == 0;
                let value = if both_empty {
                    1.0
                } else {
                    ious[p * num_gt + g]
                };
                j[(p * num_gt_ids + g) * num_timesteps + t] = value;
            }
        }
    }
    Ok(j)
}

fn bool_index_error(axis: usize, size: usize, bool_size: usize) -> MetricError {
    MetricError::Python {
        exception: "IndexError",
        message: format!(
            "boolean index did not match indexed array along axis {axis}; size of axis is \
             {size} but size of corresponding boolean axis is {bool_size}"
        ),
    }
}

fn compute_f(
    gt_dets: &[Vec<&RleMask>],
    pred_dets: &[Vec<&RleMask>],
    pred_id: usize,
    gt_id: usize,
    gt_areas: &[Vec<u32>],
    pred_areas: &[Vec<u32>],
) -> Result<Vec<f64>, MetricError> {
    let mut f = vec![0.0; gt_dets.len()];
    for (t, (gt_masks, pred_masks)) in gt_dets.iter().zip(pred_dets).enumerate() {
        let pred = pred_masks[pred_id];
        let (height, width) = (pred.height(), pred.width());
        let full_area = height as u128 * width as u128;
        let nonempty = |area: u32| 0 < area && u128::from(area) < full_area;
        let fg_nonempty = nonempty(pred_areas[t][pred_id]);
        let gt_nonempty = nonempty(gt_areas[t][gt_id]);
        if !fg_nonempty && !gt_nonempty {
            f[t] = 1.0;
            continue;
        }
        if !fg_nonempty || !gt_nonempty {
            f[t] = 0.0;
            continue;
        }
        let r = boundary_radius(height, width);
        let fg = Boundary::of(pred)?;
        let gt = Boundary::of(gt_masks[gt_id])?;
        let fg_match = count_within(&fg.points, &gt, r);
        let gt_match = count_within(&gt.points, &fg, r);
        let precision = fg_match as f64 / fg.points.len() as f64;
        let recall = gt_match as f64 / gt.points.len() as f64;
        f[t] = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
    }
    Ok(f)
}

/// numpy `linalg.norm((h, w))` is `sqrt(h*h + w*w)`, exact for sides below 2^26.
pub fn boundary_radius(height: usize, width: usize) -> i64 {
    let (h, w) = (height as f64, width as f64);
    (BOUND_TH * (h * h + w * w).sqrt()).ceil() as i64
}

pub struct Boundary {
    pub points: Vec<(i64, i64)>,
    r0: i64,
    c0: i64,
    rows: usize,
    cols: usize,
    prefix: Vec<u32>,
}

impl Boundary {
    pub fn of(mask: &RleMask) -> Result<Self, MetricError> {
        let (height, width) = (mask.height() as i64, mask.width() as i64);
        let [x, y, bw, bh] = rle::to_bbox(mask).map(|v| v as i64);
        let (r0, c0) = (0.max(y - 1), 0.max(x - 1));
        let (r1, c1) = (height.min(y + bh + 1), width.min(x + bw + 1));
        let rows = usize::try_from(r1 - r0).unwrap_or(0);
        let cols = usize::try_from(c1 - c0).unwrap_or(0);
        let seg = decode_crop(mask, r0, c0, rows, cols);
        let mut points = Vec::new();
        let mut prefix = vec![0u32; rows * (cols + 1)];
        for i in 0..rows {
            let mut count = 0u32;
            for jj in 0..cols {
                if seg2bmap_at(&seg, rows, cols, i, jj) {
                    points.push((r0 + i as i64, c0 + jj as i64));
                    count += 1;
                }
                prefix[i * (cols + 1) + jj + 1] = count;
            }
        }
        Ok(Self {
            points,
            r0,
            c0,
            rows,
            cols,
            prefix,
        })
    }

    fn any_in_row(&self, row: i64, lo: i64, hi: i64) -> bool {
        let i = row - self.r0;
        if i < 0 || i >= self.rows as i64 {
            return false;
        }
        let lo = (lo - self.c0).max(0);
        let hi = (hi - self.c0).min(self.cols as i64 - 1);
        if lo > hi {
            return false;
        }
        let base = i as usize * (self.cols + 1);
        self.prefix[base + hi as usize + 1] > self.prefix[base + lo as usize]
    }
}

fn decode_crop(mask: &RleMask, r0: i64, c0: i64, rows: usize, cols: usize) -> Vec<bool> {
    let mut seg = vec![false; rows * cols];
    let h = mask.height() as u64;
    if h == 0 || rows == 0 || cols == 0 {
        return seg;
    }
    let (r0, c0) = (r0 as u64, c0 as u64);
    let (r1, c1) = (r0 + rows as u64, c0 + cols as u64);
    let mut pos = 0u64;
    for (k, &run) in mask.runs().iter().enumerate() {
        let (start, end) = (pos, pos + u64::from(run));
        pos = end;
        if k % 2 == 0 {
            continue;
        }
        let mut at = start;
        while at < end {
            let col = at / h;
            let row = at % h;
            let seg_end = end.min((col + 1) * h);
            if (c0..c1).contains(&col) {
                let lo = row.max(r0);
                let hi = (row + (seg_end - at)).min(r1);
                for r in lo..hi {
                    seg[(r - r0) as usize * cols + (col - c0) as usize] = true;
                }
            }
            at = seg_end;
        }
    }
    seg
}

/// One pixel of TrackEval's `_seg2bmap`, with its last-row, last-column, and corner overrides.
fn seg2bmap_at(seg: &[bool], rows: usize, cols: usize, i: usize, j: usize) -> bool {
    let at = |r: usize, c: usize| seg[r * cols + c];
    let v = at(i, j);
    let last_row = i + 1 == rows;
    let last_col = j + 1 == cols;
    let e = !last_col && at(i, j + 1);
    let s = !last_row && at(i + 1, j);
    if last_row && last_col {
        false
    } else if last_col {
        v ^ s
    } else if last_row {
        v ^ e
    } else {
        let se = at(i + 1, j + 1);
        (v ^ e) | (v ^ s) | (v ^ se)
    }
}

/// TrackEval's `cv2.dilate` with `skimage.morphology.disk(r)`: matched when `dx² + dy² <= r²`.
pub fn count_within(points: &[(i64, i64)], other: &Boundary, r: i64) -> usize {
    let r2 = r * r;
    let half: Vec<i64> = (-r..=r)
        .map(|dy| ((r2 - dy * dy) as u64).isqrt() as i64)
        .collect();
    points
        .iter()
        .filter(|&&(y, x)| {
            (-r..=r)
                .zip(&half)
                .any(|(dy, &dx)| other.any_in_row(y + dy, x - dx, x + dx))
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint8_cast_wraps_like_numpy() {
        let cases = [
            (0.0, 0),
            (255.0, 255),
            (256.0, 0),
            (300.0, 44),
            (1000.0, 232),
            (-1.0, 255),
            (2_147_483_647.0, 255),
            (2_147_483_648.0, 0),
            (4_294_967_301.0, 0),
        ];
        for (x, want) in cases {
            assert_eq!(numpy_f64_to_u8(x), want, "{x}");
        }
    }

    #[test]
    fn radius_follows_the_frame_diagonal() {
        assert_eq!(boundary_radius(1080, 1920), 18);
        assert_eq!(boundary_radius(1, 1), 1);
    }
}
