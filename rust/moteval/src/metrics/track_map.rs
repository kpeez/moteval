//! TrackMAP metric: track-level average precision and recall over IoU thresholds, on all
//! tracks and on area and length subsets. Ported from `moteval/metrics/track_map.py`,
//! with all three combiners.
//!
//! Track IoU sums each pair's per-frame intersection and union. Python sums in ascending
//! frame order when the pair's frame-union set iterates in ascending order, and in the
//! set's iteration order otherwise; [`union_frame_order`] replays CPython's set hash
//! table to get that order.
//!
//! `combine_classes_det_averaged` weights each class by its `_num_dt_<label>` detection
//! count. This is the one intended divergence from TrackEval (whose det-averaged combiner
//! is a copy of the class-averaged one); it matches Python moteval.

use crate::metrics::{Metric, MetricError, Scores, Value, require_all};
use crate::model::{BoxXywh, GeometryKind, Matrix, SequenceData};
use crate::numeric::{self, EPS, maximum, minimum, py_max, py_min};

/// The public fields, in the Python `fields` order.
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

/// `LABELS`: all tracks, then the area ranges, then the length ranges.
pub const LABELS: [&str; 7] = [
    "all", "area_s", "area_m", "area_l", "time_s", "time_m", "time_l",
];

/// `AREA_RANGES = ((0**2, 32**2), (32**2, 96**2), (96**2, 1e5**2))`.
const AREA_RANGES: [(f64, f64); 3] = [(0.0, 1024.0), (1024.0, 9216.0), (9216.0, 1e10)];
/// `TIME_RANGES = ((0, 3), (3, 10), (10, 1e5))`.
const TIME_RANGES: [(f64, f64); 3] = [(0.0, 3.0), (3.0, 10.0), (10.0, 1e5)];

/// `len(IOU_THRESHOLDS)`.
pub const NUM_IOU_THRESHOLDS: usize = 10;
/// `len(RECALL_THRESHOLDS)`.
pub const NUM_RECALL_THRESHOLDS: usize = 101;

/// `IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)`.
pub fn iou_thresholds() -> [f64; NUM_IOU_THRESHOLDS] {
    numeric::arange(0.5, 0.96, 0.05)
        .expect("a finite, nonzero-step range")
        .try_into()
        .expect("np.arange(0.5, 0.96, 0.05) has 10 elements")
}

/// `RECALL_THRESHOLDS = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01) + 1))`.
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

/// `moteval.metrics.track_map.TrackMAP`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrackMap;

// ---------------------------------------------------------------------------
// CPython set iteration order
// ---------------------------------------------------------------------------

/// `sys.hash_info.modulus`: CPython hashes a non-negative int `n` to `n % (2**61 - 1)`.
const PY_HASH_MODULUS: u64 = (1 << 61) - 1;
/// `PySet_MINSIZE`.
const SET_MINSIZE: usize = 8;
/// `LINEAR_PROBES` in `Objects/setobject.c`.
const LINEAR_PROBES: usize = 9;
/// `PERTURB_SHIFT` in `Objects/setobject.c`.
const PERTURB_SHIFT: u32 = 5;

/// A CPython 3.14 `set` of non-negative ints with no deletions: the hash table of
/// `Objects/setobject.c`, slot for slot, so iteration visits keys in CPython's order.
struct PySet {
    /// `(key, hash)` per slot; `None` is an unused slot. Sets here never hold dummies.
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

    /// `set_add_entry`: linear probing over 10 slots when they fit below `mask`, then
    /// perturbed probing; a resize after the insert when the table is 3/5 full.
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
                        // found_unused
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
                    // found_active: the same int
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

    /// `set_insert_clean`: place a key known to be absent, with the same probe sequence.
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

    /// `set_table_resize`: the smallest power of two above `minused` (at least 8), with the
    /// old slots reinserted in slot order.
    fn resize(&mut self, minused: usize) {
        let mut newsize = SET_MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        if newsize == SET_MINSIZE && self.table.len() == SET_MINSIZE {
            // The small table stays; with no dummies (fill == used) nothing changes.
            debug_assert_eq!(self.fill, self.used);
            return;
        }
        let old = std::mem::replace(&mut self.table, vec![None; newsize]);
        for (key, hash) in old.into_iter().flatten() {
            Self::insert_clean(&mut self.table, key, hash);
        }
    }

    /// `set_merge_lock_held(self, other)`, the body of `self |= other` and of `set.copy`.
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

    /// `set(iterable)` over an iterable that is neither a set nor a dict (a dict's keys
    /// view): one `set_add_key` per key, in order.
    fn from_keys(keys: impl IntoIterator<Item = u64>) -> Self {
        let mut set = Self::new();
        for key in keys {
            set.add(key);
        }
        set
    }

    /// `self | other` (`set_or`): a copy of `self`, then `other` merged in.
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

/// The iteration order of `_union_frames(dt_frames, gt_frames)`, which is
/// `set(gt_frames.keys()) | set(dt_frames.keys())`, given each dict's keys in insertion
/// order. Keys are frame numbers (non-negative ints).
pub fn union_frame_order(dt_keys: &[u64], gt_keys: &[u64]) -> Vec<u64> {
    let gt = PySet::from_keys(gt_keys.iter().copied());
    let dt = PySet::from_keys(dt_keys.iter().copied());
    gt.union(&dt).iter().collect()
}

// ---------------------------------------------------------------------------
// Tracks and track IoU
// ---------------------------------------------------------------------------

/// One track of `_group_tracks`: its frames in insertion (ascending) order, with the box
/// and confidence the frame-keyed dicts hold.
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

/// `_group_tracks`: per-track views indexed `0..num_ids`, filled frame-major. A repeated id
/// within a frame overwrites the dict entry in place, as Python's dict assignment does.
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

/// `float(np.mean([box[2] * box[3] for box in f.values()])) if f else 0.0`.
fn track_area(track: &TrackView) -> f64 {
    if track.frames.is_empty() {
        return 0.0;
    }
    let areas: Vec<f64> = track.boxes.iter().map(|b| b[2] * b[3]).collect();
    numeric::mean(&areas)
}

/// `float(np.mean(list(c.values()))) if c else 0.0`.
fn track_score(track: &TrackView) -> f64 {
    if track.frames.is_empty() {
        return 0.0;
    }
    numeric::mean(&track.confidences)
}

/// `_track_iou(dt_frames, gt_frames)`: sums in the frame-union set's iteration order,
/// with builtin `min`/`max`.
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

/// `_union_set_order_is_ascending`.
fn union_set_order_is_ascending(dt: &TrackView, gt: &TrackView) -> bool {
    let dt_keys: Vec<u64> = dt.frames.iter().map(|&t| t as u64).collect();
    let gt_keys: Vec<u64> = gt.frames.iter().map(|&t| t as u64).collect();
    union_frame_order(&dt_keys, &gt_keys)
        .windows(2)
        .all(|w| w[1] > w[0])
}

/// One entry of `_track_iou_matrix`.
///
/// Python sums each pair's per-frame terms in ascending frame order with `np.minimum` and
/// `np.maximum`, where frames neither track holds add an exact `0.0` (which changes at most
/// the sign of a zero total, and no result reads that sign). A pair with a positive
/// intersection whose frame-union set does not iterate in ascending order is recomputed
/// by `_track_iou`, in set order.
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
                // d_area + gt_area - inter
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
        // No shared frame: every intersection term is 0.0, so the IoU is 0.0 whatever
        // the union is.
        return 0.0;
    }
    if intersect > 0.0 && !union_set_order_is_ascending(dt, gt) {
        return track_iou_in_set_order(dt, gt);
    }
    if union > 0.0 { intersect / union } else { 0.0 }
}

/// `_ignore_masks`: one 0/1 mask per label (`all` ignores nothing), each
/// `0 if lo - EPS <= value <= hi + EPS else 1`.
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

/// numpy's sort order for floats: NaN after every number, NaNs equal to each other.
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

/// `np.argsort(-scores, kind="mergesort")`: a stable sort by descending score.
fn argsort_descending(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&i, &j| numpy_float_cmp(-scores[i], -scores[j]));
    order
}

/// A score value's elements, read as numpy reads a bool or int array stored as floats.
fn matrix_columns(matrix: &Matrix) -> impl Iterator<Item = Vec<f64>> + '_ {
    (0..matrix.cols()).map(move |c| (0..matrix.rows()).map(|r| matrix.get(r, c)).collect())
}

/// `float(np.mean(values[values > -1])) if any else -1.0`.
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

        // Upstream's loaders presort dt tracks by descending score (mergesort, so ties
        // keep their order) before the order-dependent greedy match.
        let dt_order = argsort_descending(&dt_scores);
        let dt_tracks: Vec<TrackView> = dt_order.iter().map(|&i| dt_tracks[i].clone()).collect();
        let dt_scores: Vec<f64> = dt_order.iter().map(|&i| dt_scores[i]).collect();

        let gt_lengths: Vec<usize> = gt_tracks.iter().map(|t| t.frames.len()).collect();
        let gt_areas: Vec<f64> = gt_tracks.iter().map(track_area).collect();
        let dt_lengths: Vec<usize> = dt_tracks.iter().map(|t| t.frames.len()).collect();
        let dt_areas: Vec<f64> = dt_tracks.iter().map(track_area).collect();
        let gt_ig_masks = ignore_masks(&gt_lengths, &gt_areas);
        let dt_ig_masks = ignore_masks(&dt_lengths, &dt_areas);

        // ious[dt, gt]
        let ious: Vec<f64> = dt_tracks
            .iter()
            .flat_map(|dt| gt_tracks.iter().map(move |gt| track_iou(dt, gt)))
            .collect();

        let iou_thresholds = iou_thresholds();
        let mut scores = Scores::new();
        scores.insert(DT_SCORES, Value::Array(dt_scores));
        for (mask_idx, label) in LABELS.iter().enumerate() {
            let gt_ig_mask = &gt_ig_masks[mask_idx];
            // np.argsort(gt_ig_mask, kind="mergesort"): unignored first, each group in order
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

            // np.logical_or(dt_ig, np.logical_and(dt_m == -1, dt_ig_mask)), as 0.0/1.0
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
        // precision[thr, recall, label] = -1, recall[thr, label] = -1, det_weight = 0
        let mut precision = vec![-1.0; NUM_IOU_THRESHOLDS * NUM_RECALL_THRESHOLDS * num_labels];
        let mut recall = vec![-1.0; NUM_IOU_THRESHOLDS * num_labels];
        let mut det_weight = vec![0.0; NUM_IOU_THRESHOLDS * num_labels];

        for (lbl_idx, label) in LABELS.iter().enumerate() {
            // Python reads these four fields of every input in this order.
            require_all(all_res, DT_SCORES)?;
            require_all(all_res, &format!("{DT_MATCHES}{label}"))?;
            require_all(all_res, &format!("{DT_IGNORE}{label}"))?;
            require_all(all_res, &format!("{GT_IGNORE}{label}"))?;
            let dt_scores: Vec<f64> = all_res
                .iter()
                .flat_map(|res| res.array(DT_SCORES).iter().copied())
                .collect();
            let dt_order = argsort_descending(&dt_scores);
            // Columns are detections: concatenate per sequence, then reorder by score.
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
                // np.cumsum of the tp and fp indicators, as floats
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
                // rc = tp / num_gt
                let rc: Vec<f64> = tp.iter().map(|&x| x / num_gt as f64).collect();
                recall[thr_idx * num_labels + lbl_idx] = rc.last().copied().unwrap_or(0.0);

                // pr = tp / (fp + tp + np.spacing(1)), then right-to-left builtin max
                let mut pr: Vec<f64> = tp
                    .iter()
                    .zip(&fp)
                    .map(|(&t, &f)| t / ((f + t) + EPS))
                    .collect();
                for i in (1..num_tp).rev() {
                    pr[i - 1] = py_max(pr[i - 1], pr[i]);
                }

                // np.searchsorted(rc, RECALL_THRESHOLDS, side="left"); the first index past
                // the end raises IndexError and leaves the rest at 0.0.
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
        // Expected orders printed by CPython 3.14.8. 17 and 1 share slot 1 of an 8-slot
        // table, which is too small for linear probing, so the second key perturbs to 6.
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

    /// `trackmap_class_scores` in `dump_metric_internals.py`.
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
