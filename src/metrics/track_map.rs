use crate::metrics::{Metric, MetricError, Scores, Value, require_all};
use crate::model::{BoxXywh, GeometryKind, Matrix, SequenceData};
use crate::numeric::{self, EPS, maximum, minimum, py_max, py_min};

pub const FIELDS: &[&str] = &[
    "AP_all",
    "AP_area_s",
    "AP_area_m",
    "AP_area_l",
    "AP_time_s",
    "AP_time_m",
    "AP_time_l",
    "AR_all",
    "AR_area_s",
    "AR_area_m",
    "AR_area_l",
    "AR_time_s",
    "AR_time_m",
    "AR_time_l",
];

pub const LABELS: [&str; 7] = [
    "all", "area_s", "area_m", "area_l", "time_s", "time_m", "time_l",
];

const AREA_RANGES: [(f64, f64); 3] = [(0.0, 1024.0), (1024.0, 9216.0), (9216.0, 1e10)];
const TIME_RANGES: [(f64, f64); 3] = [(0.0, 3.0), (3.0, 10.0), (10.0, 1e5)];

pub const NUM_IOU_THRESHOLDS: usize = 10;
pub const NUM_RECALL_THRESHOLDS: usize = 101;

pub fn iou_thresholds() -> [f64; NUM_IOU_THRESHOLDS] {
    numeric::arange(0.5, 0.96, 0.05)
        .expect("a finite, nonzero-step range")
        .try_into()
        .expect("np.arange(0.5, 0.96, 0.05) has 10 elements")
}

pub fn recall_thresholds() -> [f64; NUM_RECALL_THRESHOLDS] {
    let num = numeric::round_half_even((1.00 - 0.0) / 0.01) + 1.0;
    assert_eq!(num, NUM_RECALL_THRESHOLDS as f64);
    numeric::linspace(0.0, 1.00, NUM_RECALL_THRESHOLDS, true)
        .try_into()
        .expect("101 recall thresholds")
}

const WEIGHT_PREFIX: &str = "_num_dt_";
const DT_MATCHES: &str = "_dt_matches_";
const DT_IGNORE: &str = "_dt_ignore_";
const GT_IGNORE: &str = "_gt_ignore_";
const DT_SCORES: &str = "_dt_scores";

#[derive(Clone, Copy, Debug, Default)]
pub struct TrackMap;

const PY_HASH_MODULUS: u64 = (1 << 61) - 1;
const SET_MINSIZE: usize = 8;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;

/// CPython 3.14's `set` of non-negative ints (`Objects/setobject.c`), slot for slot.
struct PySet {
    table: Vec<Option<(u64, u64)>>,
    fill: usize,
    used: usize,
}

impl PySet {
    fn new() -> Self {
        Self {
            table: vec![None; SET_MINSIZE],
            fill: 0,
            used: 0,
        }
    }

    fn mask(&self) -> usize {
        self.table.len() - 1
    }

    fn add(&mut self, key: u64) {
        let hash = key % PY_HASH_MODULUS;
        let mask = self.mask();
        let mut i = hash as usize & mask;
        let mut perturb = hash;
        loop {
            let mut probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            let mut j = i;
            loop {
                match self.table[j] {
                    None => {
                        self.fill += 1;
                        self.used += 1;
                        self.table[j] = Some((key, hash));
                        if self.fill * 5 < mask * 3 {
                            return;
                        }
                        let minused = if self.used > 50000 {
                            self.used * 2
                        } else {
                            self.used * 4
                        };
                        self.resize(minused);
                        return;
                    }
                    Some((k, h)) if h == hash && k == key => return,
                    Some(_) => {}
                }
                if probes == 0 {
                    break;
                }
                probes -= 1;
                j += 1;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
    }

    fn insert_clean(table: &mut [Option<(u64, u64)>], key: u64, hash: u64) {
        let mask = table.len() - 1;
        let mut perturb = hash;
        let mut i = hash as usize & mask;
        loop {
            if table[i].is_none() {
                table[i] = Some((key, hash));
                return;
            }
            if i + LINEAR_PROBES <= mask
                && let Some(slot) = table[i + 1..=i + LINEAR_PROBES]
                    .iter_mut()
                    .find(|slot| slot.is_none())
            {
                *slot = Some((key, hash));
                return;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
    }

    fn resize(&mut self, minused: usize) {
        let mut newsize = SET_MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        if newsize == SET_MINSIZE && self.table.len() == SET_MINSIZE {
            // CPython leaves a small table with no dummies as it is.
            debug_assert_eq!(self.fill, self.used);
            return;
        }
        let old = std::mem::replace(&mut self.table, vec![None; newsize]);
        for (key, hash) in old.into_iter().flatten() {
            Self::insert_clean(&mut self.table, key, hash);
        }
    }

    fn merge(&mut self, other: &Self) {
        if other.used == 0 {
            return;
        }
        if (self.fill + other.used) * 5 >= self.mask() * 3 {
            self.resize((self.used + other.used) * 2);
        }
        if self.fill == 0 && self.mask() == other.mask() && other.fill == other.used {
            self.table.clone_from(&other.table);
            self.fill = other.fill;
            self.used = other.used;
            return;
        }
        if self.fill == 0 {
            self.fill = other.used;
            self.used = other.used;
            for &(key, hash) in other.table.iter().flatten() {
                Self::insert_clean(&mut self.table, key, hash);
            }
            return;
        }
        for &(key, _) in other.table.iter().flatten() {
            self.add(key);
        }
    }

    fn from_keys(keys: impl IntoIterator<Item = u64>) -> Self {
        let mut set = Self::new();
        for key in keys {
            set.add(key);
        }
        set
    }

    fn union(&self, other: &Self) -> Self {
        let mut result = Self::new();
        result.merge(self);
        result.merge(other);
        result
    }

    fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.table.iter().flatten().map(|&(key, _)| key)
    }
}

/// TrackEval's `_compute_bb_track_iou` sums over `set(gt_track.keys()) | set(dt_track.keys())`.
pub fn union_frame_order(dt_keys: &[u64], gt_keys: &[u64]) -> Vec<u64> {
    let gt = PySet::from_keys(gt_keys.iter().copied());
    let dt = PySet::from_keys(dt_keys.iter().copied());
    gt.union(&dt).iter().collect()
}

#[derive(Clone, Debug, Default)]
struct TrackView {
    frames: Vec<usize>,
    boxes: Vec<BoxXywh>,
    confidences: Vec<f64>,
}

impl TrackView {
    fn get(&self, frame: usize) -> Option<&BoxXywh> {
        self.frames
            .binary_search(&frame)
            .ok()
            .map(|index| &self.boxes[index])
    }
}

fn group_tracks(
    ids_per_frame: &[Vec<usize>],
    boxes_per_frame: &[Vec<BoxXywh>],
    num_ids: usize,
    confidences_per_frame: Option<&[Vec<f64>]>,
) -> Vec<TrackView> {
    let mut tracks = vec![TrackView::default(); num_ids];
    for (t, ids) in ids_per_frame.iter().enumerate() {
        for (i, &track_id) in ids.iter().enumerate() {
            let track = &mut tracks[track_id];
            let confidence = confidences_per_frame.map_or(0.0, |c| c[t][i]);
            if track.frames.last() == Some(&t) {
                let last = track.frames.len() - 1;
                track.boxes[last] = boxes_per_frame[t][i];
                track.confidences[last] = confidence;
            } else {
                track.frames.push(t);
                track.boxes.push(boxes_per_frame[t][i]);
                track.confidences.push(confidence);
            }
        }
    }
    tracks
}

fn track_area(track: &TrackView) -> f64 {
    if track.frames.is_empty() {
        return 0.0;
    }
    let areas: Vec<f64> = track.boxes.iter().map(|b| b[2] * b[3]).collect();
    numeric::mean(&areas)
}

fn track_score(track: &TrackView) -> f64 {
    if track.frames.is_empty() {
        return 0.0;
    }
    numeric::mean(&track.confidences)
}

/// TrackEval's `_compute_bb_track_iou`: set iteration order, CPython builtin `min`/`max`.
fn track_iou_in_set_order(dt: &TrackView, gt: &TrackView) -> f64 {
    let dt_keys: Vec<u64> = dt.frames.iter().map(|&t| t as u64).collect();
    let gt_keys: Vec<u64> = gt.frames.iter().map(|&t| t as u64).collect();
    let mut intersect = 0.0;
    let mut union = 0.0;
    for t in union_frame_order(&dt_keys, &gt_keys) {
        let t = usize::try_from(t).expect("a frame index");
        match (dt.get(t), gt.get(t)) {
            (Some(&[dx, dy, dw, dh]), Some(&[gx, gy, gw, gh])) => {
                let w = py_max(py_min(dx + dw, gx + gw) - py_max(dx, gx), 0.0);
                let h = py_max(py_min(dy + dh, gy + gh) - py_max(dy, gy), 0.0);
                let i = w * h;
                union += (dw * dh + gw * gh) - i;
                intersect += i;
            }
            (None, Some(g)) => union += g[2] * g[3],
            (Some(d), None) => union += d[2] * d[3],
            (None, None) => unreachable!("a union frame is in one of the tracks"),
        }
    }
    if union > 0.0 { intersect / union } else { 0.0 }
}

fn union_set_order_is_ascending(dt: &TrackView, gt: &TrackView) -> bool {
    let dt_keys: Vec<u64> = dt.frames.iter().map(|&t| t as u64).collect();
    let gt_keys: Vec<u64> = gt.frames.iter().map(|&t| t as u64).collect();
    union_frame_order(&dt_keys, &gt_keys)
        .windows(2)
        .all(|w| w[1] > w[0])
}

fn track_iou(dt: &TrackView, gt: &TrackView) -> f64 {
    let (mut a, mut b) = (0, 0);
    let mut intersect = 0.0;
    let mut union = 0.0;
    let mut overlap = false;
    while a < dt.frames.len() || b < gt.frames.len() {
        let next_dt = dt.frames.get(a).copied();
        let next_gt = gt.frames.get(b).copied();
        match (next_dt, next_gt) {
            (Some(td), Some(tg)) if td == tg => {
                overlap = true;
                let [dx, dy, dw, dh] = dt.boxes[a];
                let [gx, gy, gw, gh] = gt.boxes[b];
                let w = maximum(minimum(dx + dw, gx + gw) - maximum(dx, gx), 0.0);
                let h = maximum(minimum(dy + dh, gy + gh) - maximum(dy, gy), 0.0);
                let inter = w * h;
                union += (dw * dh + gw * gh) - inter;
                intersect += inter;
                a += 1;
                b += 1;
            }
            (Some(td), Some(tg)) if tg < td => {
                let g = gt.boxes[b];
                union += g[2] * g[3];
                b += 1;
            }
            (Some(_), _) => {
                let d = dt.boxes[a];
                union += d[2] * d[3];
                a += 1;
            }
            (None, Some(_)) => {
                let g = gt.boxes[b];
                union += g[2] * g[3];
                b += 1;
            }
            (None, None) => unreachable!("the loop runs while a track has frames left"),
        }
    }
    if !overlap {
        return 0.0;
    }
    if intersect > 0.0 && !union_set_order_is_ascending(dt, gt) {
        return track_iou_in_set_order(dt, gt);
    }
    if union > 0.0 { intersect / union } else { 0.0 }
}

fn ignore_masks(lengths: &[usize], areas: &[f64]) -> [Vec<u8>; 7] {
    let in_range = |value: f64, (lo, hi): (f64, f64)| lo - EPS <= value && value <= hi + EPS;
    let area_mask = |range| {
        areas
            .iter()
            .map(|&a| u8::from(!in_range(a, range)))
            .collect()
    };
    let time_mask = |range| {
        lengths
            .iter()
            .map(|&n| u8::from(!in_range(n as f64, range)))
            .collect()
    };
    [
        vec![0; lengths.len()],
        area_mask(AREA_RANGES[0]),
        area_mask(AREA_RANGES[1]),
        area_mask(AREA_RANGES[2]),
        time_mask(TIME_RANGES[0]),
        time_mask(TIME_RANGES[1]),
        time_mask(TIME_RANGES[2]),
    ]
}

/// numpy's float sort order: NaN after every number.
fn numpy_float_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    let lt = |x: f64, y: f64| x < y || (y.is_nan() && !x.is_nan());
    if lt(a, b) {
        std::cmp::Ordering::Less
    } else if lt(b, a) {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

fn argsort_descending(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&i, &j| numpy_float_cmp(-scores[i], -scores[j]));
    order
}

fn matrix_columns(matrix: &Matrix) -> impl Iterator<Item = Vec<f64>> + '_ {
    (0..matrix.cols()).map(move |c| (0..matrix.rows()).map(|r| matrix.get(r, c)).collect())
}

fn mean_of_valid(values: impl Iterator<Item = f64>) -> f64 {
    let valid: Vec<f64> = values.filter(|&v| v > -1.0).collect();
    if valid.is_empty() {
        -1.0
    } else {
        numeric::mean(&valid)
    }
}

impl Metric for TrackMap {
    fn name(&self) -> &'static str {
        "TrackMAP"
    }

    fn fields(&self) -> &'static [&'static str] {
        FIELDS
    }

    fn geometry(&self) -> Option<GeometryKind> {
        Some(GeometryKind::Boxes)
    }

    fn eval_sequence(&self, data: &SequenceData) -> Result<Scores, MetricError> {
        let num_gt = data.num_gt_ids();
        let num_dt = data.num_pred_ids();
        let gt_tracks = group_tracks(data.gt_ids(), data.gt_boxes()?, num_gt, None);
        let dt_tracks = group_tracks(
            data.pred_ids(),
            data.pred_boxes()?,
            num_dt,
            Some(data.pred_confidences()),
        );
        let dt_scores: Vec<f64> = dt_tracks.iter().map(track_score).collect();

        // TrackEval's TAO loader sorts dt tracks by score (descending, mergesort).
        let dt_order = argsort_descending(&dt_scores);
        let dt_tracks: Vec<TrackView> = dt_order.iter().map(|&i| dt_tracks[i].clone()).collect();
        let dt_scores: Vec<f64> = dt_order.iter().map(|&i| dt_scores[i]).collect();

        let gt_lengths: Vec<usize> = gt_tracks.iter().map(|t| t.frames.len()).collect();
        let gt_areas: Vec<f64> = gt_tracks.iter().map(track_area).collect();
        let dt_lengths: Vec<usize> = dt_tracks.iter().map(|t| t.frames.len()).collect();
        let dt_areas: Vec<f64> = dt_tracks.iter().map(track_area).collect();
        let gt_ig_masks = ignore_masks(&gt_lengths, &gt_areas);
        let dt_ig_masks = ignore_masks(&dt_lengths, &dt_areas);

        let ious: Vec<f64> = dt_tracks
            .iter()
            .flat_map(|dt| gt_tracks.iter().map(move |gt| track_iou(dt, gt)))
            .collect();

        let iou_thresholds = iou_thresholds();
        let mut scores = Scores::new();
        scores.insert(DT_SCORES, Value::Array(dt_scores));
        for (mask_idx, label) in LABELS.iter().enumerate() {
            let gt_ig_mask = &gt_ig_masks[mask_idx];
            let mut gt_order: Vec<usize> = (0..num_gt).collect();
            gt_order.sort_by_key(|&g| gt_ig_mask[g]);
            let gt_ig: Vec<u8> = gt_order.iter().map(|&g| gt_ig_mask[g]).collect();

            let mut gt_taken = vec![false; NUM_IOU_THRESHOLDS * num_gt];
            let mut dt_m = vec![-1.0; NUM_IOU_THRESHOLDS * num_dt];
            let mut dt_ig = vec![0u8; NUM_IOU_THRESHOLDS * num_dt];

            for (thr_idx, &iou_thr) in iou_thresholds.iter().enumerate() {
                if num_dt == 0 {
                    break;
                }
                for dt_idx in 0..num_dt {
                    let mut iou = py_min(iou_thr, 1.0 - 1e-10);
                    let mut m: Option<usize> = None;
                    for gi in 0..num_gt {
                        if gt_taken[thr_idx * num_gt + gi] {
                            continue;
                        }
                        if let Some(matched) = m
                            && gt_ig[matched] == 0
                            && gt_ig[gi] == 1
                        {
                            break;
                        }
                        let candidate = ious[dt_idx * num_gt + gt_order[gi]];
                        if candidate < iou - EPS {
                            continue;
                        }
                        iou = candidate;
                        m = Some(gi);
                    }
                    let Some(m) = m else { continue };
                    dt_ig[thr_idx * num_dt + dt_idx] = gt_ig[m];
                    dt_m[thr_idx * num_dt + dt_idx] = m as f64;
                    gt_taken[thr_idx * num_gt + m] = true;
                }
            }

            let dt_ig_mask = &dt_ig_masks[mask_idx];
            let dt_ignore: Vec<f64> = (0..NUM_IOU_THRESHOLDS * num_dt)
                .map(|k| {
                    let ignored =
                        dt_ig[k] != 0 || (dt_m[k] == -1.0 && dt_ig_mask[k % num_dt.max(1)] != 0);
                    f64::from(u8::from(ignored))
                })
                .collect();

            scores.insert(
                format!("{DT_MATCHES}{label}"),
                Value::Matrix(Matrix::new(NUM_IOU_THRESHOLDS, num_dt, dt_m)?),
            );
            scores.insert(
                format!("{DT_IGNORE}{label}"),
                Value::Matrix(Matrix::new(NUM_IOU_THRESHOLDS, num_dt, dt_ignore)?),
            );
            scores.insert(
                format!("{GT_IGNORE}{label}"),
                Value::Array(gt_ig.iter().map(|&g| f64::from(g)).collect()),
            );
        }
        Ok(scores)
    }

    fn combine_sequences(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "TrackMAP" });
        }
        let recall_thresholds = recall_thresholds();
        let num_labels = LABELS.len();
        let mut precision = vec![-1.0; NUM_IOU_THRESHOLDS * NUM_RECALL_THRESHOLDS * num_labels];
        let mut recall = vec![-1.0; NUM_IOU_THRESHOLDS * num_labels];
        let mut det_weight = vec![0.0; NUM_IOU_THRESHOLDS * num_labels];

        for (lbl_idx, label) in LABELS.iter().enumerate() {
            require_all(all_res, DT_SCORES)?;
            require_all(all_res, &format!("{DT_MATCHES}{label}"))?;
            require_all(all_res, &format!("{DT_IGNORE}{label}"))?;
            require_all(all_res, &format!("{GT_IGNORE}{label}"))?;
            let dt_scores: Vec<f64> = all_res
                .iter()
                .flat_map(|res| res.array(DT_SCORES).iter().copied())
                .collect();
            let dt_order = argsort_descending(&dt_scores);
            let concat_columns = |field: &str| -> Vec<Vec<f64>> {
                let columns: Vec<Vec<f64>> = all_res
                    .iter()
                    .flat_map(|res| matrix_columns(res.matrix(field)).collect::<Vec<_>>())
                    .collect();
                dt_order.iter().map(|&i| columns[i].clone()).collect()
            };
            let dt_m = concat_columns(&format!("{DT_MATCHES}{label}"));
            let dt_ig = concat_columns(&format!("{DT_IGNORE}{label}"));
            let num_gt = all_res
                .iter()
                .flat_map(|res| res.array(&format!("{GT_IGNORE}{label}")).iter())
                .filter(|&&g| g == 0.0)
                .count();
            if num_gt == 0 {
                continue;
            }
            let num_tp = dt_m.len();

            for thr_idx in 0..NUM_IOU_THRESHOLDS {
                let mut tp = Vec::with_capacity(num_tp);
                let mut fp = Vec::with_capacity(num_tp);
                let (mut tp_sum, mut fp_sum) = (0u64, 0u64);
                for k in 0..num_tp {
                    let matched = dt_m[k][thr_idx] != -1.0;
                    let ignored = dt_ig[k][thr_idx] != 0.0;
                    tp_sum += u64::from(matched && !ignored);
                    fp_sum += u64::from(!matched && !ignored);
                    tp.push(tp_sum as f64);
                    fp.push(fp_sum as f64);
                }
                let rc: Vec<f64> = tp.iter().map(|&x| x / num_gt as f64).collect();
                recall[thr_idx * num_labels + lbl_idx] = rc.last().copied().unwrap_or(0.0);

                // TrackEval makes precision monotonic from right to left.
                let mut pr: Vec<f64> = tp
                    .iter()
                    .zip(&fp)
                    .map(|(&t, &f)| t / ((f + t) + EPS))
                    .collect();
                for i in (1..num_tp).rev() {
                    pr[i - 1] = py_max(pr[i - 1], pr[i]);
                }

                // TrackEval: an index past the end raises `IndexError`, so the rest stay 0.0.
                let mut pr_at_recall = [0.0; NUM_RECALL_THRESHOLDS];
                for (r_idx, &threshold) in recall_thresholds.iter().enumerate() {
                    let pr_idx = rc.partition_point(|&x| x < threshold);
                    let Some(&value) = pr.get(pr_idx) else { break };
                    pr_at_recall[r_idx] = value;
                }
                for (r_idx, &value) in pr_at_recall.iter().enumerate() {
                    precision[(thr_idx * NUM_RECALL_THRESHOLDS + r_idx) * num_labels + lbl_idx] =
                        value;
                }
                det_weight[thr_idx * num_labels + lbl_idx] =
                    dt_ig.iter().filter(|col| col[thr_idx] == 0.0).count() as f64;
            }
        }

        let mut scores = Scores::new();
        for (lbl_idx, label) in LABELS.iter().enumerate() {
            let ap: Vec<f64> =
                (0..NUM_IOU_THRESHOLDS)
                    .map(|a| {
                        mean_of_valid((0..NUM_RECALL_THRESHOLDS).map(|r| {
                            precision[(a * NUM_RECALL_THRESHOLDS + r) * num_labels + lbl_idx]
                        }))
                    })
                    .collect();
            let ar: Vec<f64> = (0..NUM_IOU_THRESHOLDS)
                .map(|a| recall[a * num_labels + lbl_idx])
                .collect();
            let weight: Vec<f64> = (0..NUM_IOU_THRESHOLDS)
                .map(|a| det_weight[a * num_labels + lbl_idx])
                .collect();
            scores.insert(format!("AP_{label}"), Value::Array(ap));
            scores.insert(format!("AR_{label}"), Value::Array(ar));
            scores.insert(format!("{WEIGHT_PREFIX}{label}"), Value::Array(weight));
        }
        Ok(scores)
    }

    fn combine_classes_class_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "TrackMAP" });
        }
        let mut scores = Scores::new();
        for field in FIELDS {
            require_all(all_res, field)?;
            let out: Vec<f64> = (0..NUM_IOU_THRESHOLDS)
                .map(|a| mean_of_valid(all_res.iter().map(|res| res.array(field)[a])))
                .collect();
            scores.insert(*field, Value::Array(out));
        }
        Ok(scores)
    }

    fn combine_classes_det_averaged(&self, all_res: &[&Scores]) -> Result<Scores, MetricError> {
        // Intended divergence: TrackEval reuses its class mean; this weights by `_num_dt_*`.
        if all_res.is_empty() {
            return Err(MetricError::NoResults { metric: "TrackMAP" });
        }
        let mut scores = Scores::new();
        for label in LABELS {
            let weight_field = format!("{WEIGHT_PREFIX}{label}");
            require_all(all_res, &weight_field)?;
            for field in [format!("AP_{label}"), format!("AR_{label}")] {
                require_all(all_res, &field)?;
                let mut out = Vec::with_capacity(NUM_IOU_THRESHOLDS);
                for a in 0..NUM_IOU_THRESHOLDS {
                    let mut values = Vec::new();
                    let mut weights = Vec::new();
                    for res in all_res {
                        let value = res.array(&field)[a];
                        if value > -1.0 {
                            values.push(value);
                            weights.push(res.array(&weight_field)[a]);
                        }
                    }
                    out.push(if values.is_empty() {
                        -1.0
                    } else if numeric::sum(&weights) > 0.0 {
                        numeric::average(&values, &weights)?
                    } else {
                        numeric::mean(&values)
                    });
                }
                scores.insert(field, Value::Array(out));
            }
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_sets_iterate_by_slot() {
        // Expected orders printed by CPython 3.14.8.
        assert_eq!(union_frame_order(&[], &[17, 1]), [17, 1]);
        assert_eq!(union_frame_order(&[1], &[17]), [17, 1]);
        assert_eq!(union_frame_order(&[3, 2], &[]), [2, 3]);
        assert_eq!(
            union_frame_order(&[1, 9, 17, 25, 33], &[0, 8, 16]),
            [0, 1, 33, 8, 9, 16, 17, 25]
        );
        assert_eq!(
            union_frame_order(
                &[100, 101, 102, 103, 104, 105],
                &[98, 99, 100, 101, 102, 103]
            ),
            [98, 99, 100, 101, 102, 103, 104, 105]
        );
    }

    #[test]
    fn constants_match_numpy_lengths() {
        assert_eq!(iou_thresholds()[0], 0.5);
        let recalls = recall_thresholds();
        assert_eq!((recalls[0], recalls[100]), (0.0, 1.0));
    }

    use crate::metrics::hota::internals_testdata::{
        SplitMix64, digest, expected_digest, fnv1a64_words, load, scores_inputs_fnv, u64_of,
    };

    fn class_scores(rng: &mut SplitMix64) -> Scores {
        let mut scores = Scores::new();
        for label in LABELS {
            for field in [format!("AP_{label}"), format!("AR_{label}")] {
                let values = (0..NUM_IOU_THRESHOLDS)
                    .map(|_| {
                        if rng.below(4) == 0 {
                            -1.0
                        } else {
                            rng.uniform()
                        }
                    })
                    .collect();
                scores.insert(field, Value::Array(values));
            }
            let weights = (0..NUM_IOU_THRESHOLDS)
                .map(|_| {
                    let a = rng.below(4);
                    (a * rng.below(30)) as f64
                })
                .collect();
            scores.insert(format!("{WEIGHT_PREFIX}{label}"), Value::Array(weights));
        }
        scores
    }

    #[test]
    fn class_combiners_match_python_with_many_classes() {
        let json = load();
        let cases = json["trackmap_class_combiners"].as_array().unwrap();
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
            let class_averaged = TrackMap.combine_classes_class_averaged(&refs).unwrap();
            let det_averaged = TrackMap.combine_classes_det_averaged(&refs).unwrap();
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

    #[test]
    fn track_lengths_areas_and_scores_match_python() {
        let json = load();
        let cases = json["trackmap_tracks"].as_array().unwrap();
        assert_eq!(cases.len(), 300);
        for case in cases {
            let seed = u64_of(&case["seed"]);
            let mut rng = SplitMix64::new(seed);
            let num_timesteps = 20 + rng.below(40);
            let num_ids = 1 + rng.below(6) as usize;
            let (mut ids, mut boxes, mut confidences) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..num_timesteps {
                let (mut frame_ids, mut frame_boxes, mut frame_confs) =
                    (Vec::new(), Vec::new(), Vec::new());
                for track_id in 0..num_ids {
                    if rng.below(3) == 0 {
                        continue;
                    }
                    frame_ids.push(track_id);
                    let x = rng.uniform() * 50.0;
                    let y = rng.uniform() * 50.0;
                    let w = rng.uniform() * 64.0;
                    let h = rng.uniform() * 64.0;
                    frame_boxes.push([x, y, w, h]);
                    frame_confs.push(rng.uniform());
                }
                ids.push(frame_ids);
                boxes.push(frame_boxes);
                confidences.push(frame_confs);
            }
            let mut inputs = Vec::new();
            for ((frame_ids, frame_boxes), frame_confs) in ids.iter().zip(&boxes).zip(&confidences)
            {
                inputs.push(frame_ids.len() as u64);
                inputs.extend(frame_ids.iter().map(|&id| id as u64));
                inputs.extend(frame_boxes.iter().flatten().map(|x| x.to_bits()));
                inputs.extend(frame_confs.iter().map(|x| x.to_bits()));
            }
            assert_eq!(
                fnv1a64_words(inputs),
                u64_of(&case["inputs_fnv"]),
                "seed {seed}: generator drift"
            );
            let tracks = group_tracks(&ids, &boxes, num_ids, Some(&confidences));
            let lengths: Vec<u64> = tracks.iter().map(|t| t.frames.len() as u64).collect();
            let areas: Vec<u64> = tracks.iter().map(|t| track_area(t).to_bits()).collect();
            let scores: Vec<u64> = tracks.iter().map(|t| track_score(t).to_bits()).collect();
            let want = |key: &str| -> Vec<u64> {
                case[key].as_array().unwrap().iter().map(u64_of).collect()
            };
            assert_eq!(lengths, want("lengths"), "seed {seed}: lengths");
            assert_eq!(areas, want("areas"), "seed {seed}: areas");
            assert_eq!(scores, want("scores"), "seed {seed}: scores");
        }
    }
}
