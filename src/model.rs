use std::fmt;

use crate::formats::{MaskTrack, Track};
use crate::pyfmt::repr_str;
use crate::rle::RleMask;

pub type BoxXywh = [f64; 4];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    FrameOutOfRange {
        frame: i64,
        convention: String,
        first_frame: i64,
        num_timesteps: usize,
    },
    WrongGeometry {
        sequence: String,
        holds: GeometryKind,
    },
    Inconsistent(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameOutOfRange {
                frame,
                convention,
                first_frame,
                num_timesteps,
            } => {
                let stop = *first_frame as i128 + *num_timesteps as i128;
                write!(
                    f,
                    "frame {frame} is out of range for convention {} (first_frame={first_frame}); \
                     valid frames are [{first_frame}, {}]",
                    repr_str(convention),
                    stop - 1
                )
            }
            Self::WrongGeometry { sequence, holds } => {
                let (holds, wanted) = match holds {
                    GeometryKind::Masks => ("mask", "boxes"),
                    GeometryKind::Boxes => ("box", "masks"),
                };
                write!(
                    f,
                    "sequence {} carries {holds} geometry, not {wanted}",
                    repr_str(sequence)
                )
            }
            Self::Inconsistent(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameConvention {
    pub name: String,
    pub first_frame: i64,
}

impl FrameConvention {
    pub fn new(name: impl Into<String>, first_frame: i64) -> Self {
        Self {
            name: name.into(),
            first_frame,
        }
    }

    pub fn to_index(&self, frame: i64, num_timesteps: usize) -> Result<usize, ModelError> {
        let index = frame as i128 - self.first_frame as i128;
        if index < 0 || index >= num_timesteps as i128 {
            return Err(ModelError::FrameOutOfRange {
                frame,
                convention: self.name.clone(),
                first_frame: self.first_frame,
                num_timesteps,
            });
        }
        Ok(index as usize)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Protocol {
    pub name: String,
    pub frame_convention: FrameConvention,
    pub eval_classes: Vec<i64>,
    pub distractor_classes: Vec<i64>,
    pub distractor_iou_threshold: f64,
    pub ignore_iou_threshold: f64,
    pub drop_zero_conf_gt: bool,
    pub matching_fill: f64,
}

impl Protocol {
    pub fn new(
        name: impl Into<String>,
        frame_convention: FrameConvention,
        eval_classes: Vec<i64>,
    ) -> Self {
        Self {
            name: name.into(),
            frame_convention,
            eval_classes,
            distractor_classes: Vec::new(),
            distractor_iou_threshold: 0.5,
            ignore_iou_threshold: 0.5,
            drop_zero_conf_gt: true,
            matching_fill: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GtSequence {
    pub name: String,
    pub num_timesteps: usize,
    pub tracks: Vec<Track>,
    pub ignore_regions: Vec<Track>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskGtSequence {
    pub name: String,
    pub num_timesteps: usize,
    pub tracks: Vec<MaskTrack>,
    pub ignore_regions: Vec<MaskTrack>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Sequence {
    Boxes(GtSequence),
    Masks(MaskGtSequence),
}

impl Sequence {
    pub fn name(&self) -> &str {
        match self {
            Self::Boxes(seq) => &seq.name,
            Self::Masks(seq) => &seq.name,
        }
    }

    pub fn num_timesteps(&self) -> usize {
        match self {
            Self::Boxes(seq) => seq.num_timesteps,
            Self::Masks(seq) => seq.num_timesteps,
        }
    }

    pub fn kind(&self) -> GeometryKind {
        match self {
            Self::Boxes(_) => GeometryKind::Boxes,
            Self::Masks(_) => GeometryKind::Masks,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MOTDataset {
    pub name: String,
    pub split: String,
    pub sequences: Vec<Sequence>,
    pub protocol: Protocol,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeometryKind {
    Boxes,
    Masks,
}

impl GeometryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Boxes => "boxes",
            Self::Masks => "masks",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, ModelError> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(ModelError::Inconsistent(format!(
                "matrix data has {} values, not {rows} x {cols}",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::filled(rows, cols, 0.0)
    }

    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f64> {
        self.data
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(
            row < self.rows && col < self.cols,
            "matrix index out of range"
        );
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(
            row < self.rows && col < self.cols,
            "matrix index out of range"
        );
        self.data[row * self.cols + col] = value;
    }

    pub fn row(&self, row: usize) -> &[f64] {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

#[derive(Clone, Debug)]
pub enum Geometry {
    Boxes {
        gt: Vec<Vec<BoxXywh>>,
        pred: Vec<Vec<BoxXywh>>,
    },
    Masks {
        gt: Vec<Vec<RleMask>>,
        pred: Vec<Vec<RleMask>>,
    },
}

impl Geometry {
    pub fn kind(&self) -> GeometryKind {
        match self {
            Self::Boxes { .. } => GeometryKind::Boxes,
            Self::Masks { .. } => GeometryKind::Masks,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SequenceDataParts {
    pub name: String,
    pub num_timesteps: usize,
    pub num_gt_ids: usize,
    pub num_pred_ids: usize,
    pub num_gt_dets: usize,
    pub num_pred_dets: usize,
    pub gt_ids: Vec<Vec<usize>>,
    pub pred_ids: Vec<Vec<usize>>,
    pub pred_confidences: Vec<Vec<f64>>,
    pub geometry: Geometry,
    pub similarity: Vec<Matrix>,
}

#[derive(Clone, Debug)]
pub struct SequenceData {
    parts: SequenceDataParts,
}

impl SequenceData {
    pub fn new(parts: SequenceDataParts) -> Result<Self, ModelError> {
        let n = parts.num_timesteps;
        let bad = |what: &str| {
            Err(ModelError::Inconsistent(format!(
                "sequence {}: {what}",
                repr_str(&parts.name)
            )))
        };
        let (geo_gt_len, geo_pred_len) = match &parts.geometry {
            Geometry::Boxes { gt, pred } => (gt.len(), pred.len()),
            Geometry::Masks { gt, pred } => (gt.len(), pred.len()),
        };
        if parts.gt_ids.len() != n
            || parts.pred_ids.len() != n
            || parts.pred_confidences.len() != n
            || parts.similarity.len() != n
            || geo_gt_len != n
            || geo_pred_len != n
        {
            return bad("a per-frame field does not have num_timesteps entries");
        }
        let mut gt_dets = 0;
        let mut pred_dets = 0;
        for t in 0..n {
            let (num_gt, num_pred) = (parts.gt_ids[t].len(), parts.pred_ids[t].len());
            gt_dets += num_gt;
            pred_dets += num_pred;
            if parts.gt_ids[t].iter().any(|&id| id >= parts.num_gt_ids)
                || parts.pred_ids[t].iter().any(|&id| id >= parts.num_pred_ids)
            {
                return bad("an id is not below its num_*_ids");
            }
            if parts.pred_confidences[t].len() != num_pred {
                return bad("pred_confidences and pred_ids differ in length");
            }
            if parts.similarity[t].shape() != (num_gt, num_pred) {
                return bad("a similarity matrix is not num_gt x num_pred");
            }
            let (geo_gt, geo_pred) = match &parts.geometry {
                Geometry::Boxes { gt, pred } => (gt[t].len(), pred[t].len()),
                Geometry::Masks { gt, pred } => (gt[t].len(), pred[t].len()),
            };
            if geo_gt != num_gt || geo_pred != num_pred {
                return bad("geometry and ids differ in length");
            }
        }
        if gt_dets != parts.num_gt_dets || pred_dets != parts.num_pred_dets {
            return bad("num_gt_dets or num_pred_dets does not count the detections");
        }
        Ok(Self { parts })
    }

    pub fn name(&self) -> &str {
        &self.parts.name
    }

    pub fn num_timesteps(&self) -> usize {
        self.parts.num_timesteps
    }

    pub fn num_gt_ids(&self) -> usize {
        self.parts.num_gt_ids
    }

    pub fn num_pred_ids(&self) -> usize {
        self.parts.num_pred_ids
    }

    pub fn num_gt_dets(&self) -> usize {
        self.parts.num_gt_dets
    }

    pub fn num_pred_dets(&self) -> usize {
        self.parts.num_pred_dets
    }

    pub fn gt_ids(&self) -> &[Vec<usize>] {
        &self.parts.gt_ids
    }

    pub fn pred_ids(&self) -> &[Vec<usize>] {
        &self.parts.pred_ids
    }

    pub fn pred_confidences(&self) -> &[Vec<f64>] {
        &self.parts.pred_confidences
    }

    pub fn similarity(&self) -> &[Matrix] {
        &self.parts.similarity
    }

    pub fn geometry(&self) -> &Geometry {
        &self.parts.geometry
    }

    pub fn gt_boxes(&self) -> Result<&[Vec<BoxXywh>], ModelError> {
        match &self.parts.geometry {
            Geometry::Boxes { gt, .. } => Ok(gt),
            Geometry::Masks { .. } => Err(self.wrong_geometry()),
        }
    }

    pub fn pred_boxes(&self) -> Result<&[Vec<BoxXywh>], ModelError> {
        match &self.parts.geometry {
            Geometry::Boxes { pred, .. } => Ok(pred),
            Geometry::Masks { .. } => Err(self.wrong_geometry()),
        }
    }

    pub fn gt_masks(&self) -> Result<&[Vec<RleMask>], ModelError> {
        match &self.parts.geometry {
            Geometry::Masks { gt, .. } => Ok(gt),
            Geometry::Boxes { .. } => Err(self.wrong_geometry()),
        }
    }

    pub fn pred_masks(&self) -> Result<&[Vec<RleMask>], ModelError> {
        match &self.parts.geometry {
            Geometry::Masks { pred, .. } => Ok(pred),
            Geometry::Boxes { .. } => Err(self.wrong_geometry()),
        }
    }

    pub fn into_parts(self) -> SequenceDataParts {
        self.parts
    }

    fn wrong_geometry(&self) -> ModelError {
        ModelError::WrongGeometry {
            sequence: self.parts.name.clone(),
            holds: self.parts.geometry.kind(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_out_of_range_message_matches_python() {
        let convention = FrameConvention::new("1-indexed", 1);
        assert_eq!(convention.to_index(1, 5), Ok(0));
        assert_eq!(convention.to_index(5, 5), Ok(4));
        let err = convention.to_index(6, 5).unwrap_err();
        assert_eq!(
            err.to_string(),
            "frame 6 is out of range for convention '1-indexed' (first_frame=1); \
             valid frames are [1, 5]"
        );
        let err = FrameConvention::new("0-indexed", 0)
            .to_index(-1, 3)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "frame -1 is out of range for convention '0-indexed' (first_frame=0); \
             valid frames are [0, 2]"
        );
    }

    #[test]
    fn wrong_geometry_message_matches_python() {
        let err = ModelError::WrongGeometry {
            sequence: "seq-01".into(),
            holds: GeometryKind::Masks,
        };
        assert_eq!(
            err.to_string(),
            "sequence 'seq-01' carries mask geometry, not boxes"
        );
    }

    #[test]
    fn sequence_data_rejects_inconsistent_parts() {
        let parts = SequenceDataParts {
            name: "s".into(),
            num_timesteps: 1,
            num_gt_ids: 1,
            num_pred_ids: 0,
            num_gt_dets: 1,
            num_pred_dets: 0,
            gt_ids: vec![vec![0]],
            pred_ids: vec![vec![]],
            pred_confidences: vec![vec![]],
            geometry: Geometry::Boxes {
                gt: vec![vec![[0.0, 0.0, 1.0, 1.0]]],
                pred: vec![vec![]],
            },
            similarity: vec![Matrix::zeros(1, 0)],
        };
        assert!(SequenceData::new(parts.clone()).is_ok());
        let mut bad = parts;
        bad.similarity = vec![Matrix::zeros(0, 0)];
        assert!(SequenceData::new(bad).is_err());
    }
}
