pub mod testgen;

use std::fs;
use std::path::{Path, PathBuf};

use moteval::metrics::{Metric, Scores, Value};
use moteval::model::{Geometry, Matrix, SequenceData, SequenceDataParts};
use moteval::rle::RleMask;
use serde_json::Value as Json;

pub fn testdata_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/integration/data")
}

pub fn read_json(path: &Path) -> Json {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("cannot parse {}: {err}", path.display()))
}

pub fn as_u64(json: &Json) -> u64 {
    json.as_u64()
        .unwrap_or_else(|| panic!("expected an unsigned integer, got {json}"))
}

pub fn as_usize(json: &Json) -> usize {
    usize::try_from(as_u64(json)).expect("fits usize")
}

pub fn as_i64(json: &Json) -> i64 {
    json.as_i64()
        .unwrap_or_else(|| panic!("expected an integer, got {json}"))
}

pub fn as_str(json: &Json) -> &str {
    json.as_str()
        .unwrap_or_else(|| panic!("expected a string, got {json}"))
}

pub fn as_array(json: &Json) -> &Vec<Json> {
    json.as_array()
        .unwrap_or_else(|| panic!("expected an array, got {json}"))
}

pub fn f64_from_bits(json: &Json) -> f64 {
    f64::from_bits(as_u64(json))
}

pub fn f64s_from_bits(json: &Json) -> Vec<f64> {
    as_array(json).iter().map(f64_from_bits).collect()
}

pub fn usizes(json: &Json) -> Vec<usize> {
    as_array(json).iter().map(as_usize).collect()
}

pub fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

pub fn assert_f64s_bit_equal(expected: &[f64], got: &[f64], label: &str) {
    assert_eq!(
        expected.len(),
        got.len(),
        "{label}: length {} != expected {}",
        got.len(),
        expected.len()
    );
    for (i, (&e, &g)) in expected.iter().zip(got).enumerate() {
        assert!(
            same_bits(e, g),
            "{label}[{i}]: got {g:e} ({:#018x}), expected {e:e} ({:#018x})",
            g.to_bits(),
            e.to_bits()
        );
    }
}

pub fn decode_value(json: &Json) -> Value {
    let shape = usizes(&json["shape"]);
    let data = f64s_from_bits(&json["bits"]);
    match shape.as_slice() {
        [] => {
            assert_eq!(data.len(), 1, "scalar with {} values", data.len());
            Value::Float(data[0])
        }
        [n] => {
            assert_eq!(data.len(), *n);
            Value::Array(data)
        }
        [rows, cols] => Value::Matrix(Matrix::new(*rows, *cols, data).expect("shape")),
        other => panic!("unsupported score shape {other:?}"),
    }
}

pub fn decode_scores(json: &Json) -> Scores {
    let mut scores = Scores::new();
    for pair in as_array(json) {
        scores.insert(as_str(&pair[0]), decode_value(&pair[1]));
    }
    scores
}

pub fn decode_metric_scores(json: &Json) -> Vec<(String, Scores)> {
    as_array(json)
        .iter()
        .map(|pair| (as_str(&pair[0]).to_string(), decode_scores(&pair[1])))
        .collect()
}

pub fn assert_scores_bit_equal(expected: &Scores, got: &Scores, label: &str) {
    for (field, want) in expected.iter() {
        let have = got
            .get(field)
            .unwrap_or_else(|| panic!("{label}.{field} missing from the Rust scores"));
        assert_eq!(have.shape(), want.shape(), "{label}.{field}: shape differs");
        assert_f64s_bit_equal(
            want.as_slice(),
            have.as_slice(),
            &format!("{label}.{field}"),
        );
    }
}

pub fn assert_field_order(expected: &Scores, got: &Scores, label: &str) {
    let want: Vec<&str> = expected.iter().map(|(f, _)| f).collect();
    let have: Vec<&str> = got.iter().map(|(f, _)| f).collect();
    assert_eq!(have, want, "{label}: field order differs from Python");
}

pub struct CaseSequence {
    pub key: String,
    pub data: SequenceData,
}

pub struct CaseView {
    pub class: Option<i64>,
    pub sequences: Vec<CaseSequence>,
    pub moteval_per_sequence: Vec<(String, Vec<(String, Scores)>)>,
    pub moteval_combined: Vec<(String, Scores)>,
}

pub struct Case {
    pub fixture: String,
    pub case: String,
    pub procedure: String,
    pub metrics: Vec<String>,
    pub views: Vec<CaseView>,
    pub oracle: Vec<(String, Vec<(String, Scores)>)>,
    pub moteval_class_averaged: Vec<(String, Scores)>,
    pub moteval_det_averaged: Vec<(String, Scores)>,
}

impl Case {
    pub fn oracle_section(&self, section: &str) -> Option<&[(String, Scores)]> {
        self.oracle
            .iter()
            .find(|(name, _)| name == section)
            .map(|(_, by_metric)| by_metric.as_slice())
    }
}

pub fn load_synthetic_case(fixture: &str, case: &str) -> Case {
    let path = testdata_dir()
        .join("scenarios")
        .join(fixture)
        .join(format!("{case}.json"));
    load_case(&path)
}

pub fn synthetic_case_names(fixture: &str) -> Vec<String> {
    let dir = testdata_dir().join("scenarios").join(fixture);
    let mut names: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("cannot list {}: {err}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .map(|p| p.file_stem().expect("stem").to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

pub fn real_testdata_dir(what: &str) -> Option<PathBuf> {
    match std::env::var_os("MOTEVAL_REAL_TESTDATA") {
        Some(dir) if !dir.is_empty() && Path::new(&dir).is_dir() => Some(PathBuf::from(dir)),
        Some(dir) => {
            eprintln!(
                "SKIPPING REAL-DATA GATE: {what}: MOTEVAL_REAL_TESTDATA={} is not a directory \
                 - generate it with `scripts/testdata/dump_scenarios.py --real-data`",
                Path::new(&dir).display()
            );
            None
        }
        None => {
            eprintln!(
                "SKIPPING REAL-DATA GATE: {what}: MOTEVAL_REAL_TESTDATA is not set \
                 - generate the data with `scripts/testdata/dump_scenarios.py --real-data`"
            );
            None
        }
    }
}

pub fn load_real_case(root: &Path, fixture: &str, case: &str) -> Option<Case> {
    let path = root.join(fixture).join(case).join("case.json");
    if !path.is_file() {
        eprintln!(
            "SKIPPING REAL-DATA GATE: {fixture}/{case}: {} not found - generate it with \
             `scripts/testdata/dump_scenarios.py --real-data`",
            path.display()
        );
        return None;
    }
    Some(load_case(&path))
}

pub fn real_case_names(root: &Path, fixture: &str) -> Vec<String> {
    let dir = root.join(fixture);
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.join("case.json").is_file())
        .map(|p| p.file_name().expect("name").to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

pub fn load_case(path: &Path) -> Case {
    let json = read_json(path);
    assert_eq!(
        as_u64(&json["format"]),
        1,
        "unknown case format in {}",
        path.display()
    );
    let dir = path.parent().expect("case file has a directory");
    let moteval_views = as_array(&json["moteval"]["views"]);
    let views = as_array(&json["views"])
        .iter()
        .zip(moteval_views)
        .map(|(view, ours)| {
            let sequences = as_array(&view["sequences"])
                .iter()
                .map(|record| load_sequence(dir, record))
                .collect();
            let moteval_per_sequence = as_array(&ours["per_sequence"])
                .iter()
                .map(|pair| (as_str(&pair[0]).to_string(), decode_metric_scores(&pair[1])))
                .collect();
            CaseView {
                class: view["class"].as_i64(),
                sequences,
                moteval_per_sequence,
                moteval_combined: decode_metric_scores(&ours["combined"]),
            }
        })
        .collect();
    let oracle = json["oracle"]
        .as_object()
        .expect("oracle sections")
        .iter()
        .map(|(section, by_metric)| (section.clone(), decode_metric_scores(by_metric)))
        .collect();
    Case {
        fixture: as_str(&json["fixture"]).to_string(),
        case: as_str(&json["case"]).to_string(),
        procedure: as_str(&json["procedure"]).to_string(),
        metrics: as_array(&json["metrics"])
            .iter()
            .map(|m| as_str(m).to_string())
            .collect(),
        views,
        oracle,
        moteval_class_averaged: decode_metric_scores(&json["moteval"]["class_averaged"]),
        moteval_det_averaged: decode_metric_scores(&json["moteval"]["det_averaged"]),
    }
}

struct FlatArrays<'a> {
    arrays: &'a serde_json::Map<String, Json>,
    blob: Option<Vec<u8>>,
}

impl FlatArrays<'_> {
    fn bytes_of(&self, name: &str, width: usize) -> &[u8] {
        let spec = &self.arrays[name];
        let blob = self
            .blob
            .as_ref()
            .expect("binary arrays need their .bin file");
        let offset = as_usize(&spec["offset"]);
        let len = as_usize(&spec["len"]) * width;
        &blob[offset..offset + len]
    }

    fn f64s(&self, name: &str) -> Vec<f64> {
        let spec = &self.arrays[name];
        assert_eq!(as_str(&spec["dtype"]), "f64", "{name} is not f64");
        if let Some(data) = spec.get("data") {
            return f64s_from_bits(data);
        }
        let (words, rest) = self.bytes_of(name, 8).as_chunks::<8>();
        assert!(
            rest.is_empty(),
            "{name}: not a whole number of 8-byte values"
        );
        words.iter().map(|b| f64::from_le_bytes(*b)).collect()
    }

    fn i64s(&self, name: &str) -> Vec<i64> {
        let spec = &self.arrays[name];
        assert_eq!(as_str(&spec["dtype"]), "i64", "{name} is not i64");
        if let Some(data) = spec.get("data") {
            return as_array(data).iter().map(as_i64).collect();
        }
        let (words, rest) = self.bytes_of(name, 8).as_chunks::<8>();
        assert!(
            rest.is_empty(),
            "{name}: not a whole number of 8-byte values"
        );
        words.iter().map(|b| i64::from_le_bytes(*b)).collect()
    }

    fn bytes(&self, name: &str) -> Vec<u8> {
        let spec = &self.arrays[name];
        match as_str(&spec["dtype"]) {
            "ascii" => as_str(&spec["data"]).as_bytes().to_vec(),
            "u8" => self.bytes_of(name, 1).to_vec(),
            other => panic!("{name}: unexpected byte dtype {other}"),
        }
    }

    fn usizes(&self, name: &str) -> Vec<usize> {
        self.i64s(name)
            .into_iter()
            .map(|x| usize::try_from(x).expect("non-negative"))
            .collect()
    }
}

fn split_frames<T: Clone>(flat: &[T], counts: &[usize], width: usize) -> Vec<Vec<T>> {
    let mut out = Vec::with_capacity(counts.len());
    let mut at = 0;
    for &n in counts {
        out.push(flat[at..at + n * width].to_vec());
        at += n * width;
    }
    assert_eq!(
        at,
        flat.len(),
        "per-frame counts do not cover the flat array"
    );
    out
}

fn masks(arrays: &FlatArrays<'_>, side: &str, counts: &[usize]) -> Vec<Vec<RleMask>> {
    let heights = arrays.usizes(&format!("{side}_mask_heights"));
    let widths = arrays.usizes(&format!("{side}_mask_widths"));
    let lens = arrays.usizes(&format!("{side}_mask_count_lens"));
    let bytes = arrays.bytes(&format!("{side}_mask_counts"));
    let mut flat = Vec::with_capacity(lens.len());
    let mut at = 0;
    for ((&h, &w), &len) in heights.iter().zip(&widths).zip(&lens) {
        flat.push(RleMask::from_compressed(h, w, &bytes[at..at + len]).expect("valid RLE"));
        at += len;
    }
    assert_eq!(
        at,
        bytes.len(),
        "mask count lengths do not cover the counts bytes"
    );
    split_frames(&flat, counts, 1)
}

fn load_sequence(dir: &Path, record: &Json) -> CaseSequence {
    let header;
    let record = match record.get("header") {
        Some(name) => {
            header = read_json(&dir.join(as_str(name)));
            &header
        }
        None => record,
    };
    let blob = record
        .get("bin")
        .map(|name| fs::read(dir.join(as_str(name))).expect("read .bin"));
    let arrays = FlatArrays {
        arrays: record["arrays"].as_object().expect("arrays"),
        blob,
    };
    let gt_counts = arrays.usizes("gt_counts");
    let pred_counts = arrays.usizes("pred_counts");
    let num_timesteps = as_usize(&record["num_timesteps"]);
    let similarity_flat = arrays.f64s("similarity");
    let mut similarity = Vec::with_capacity(num_timesteps);
    let mut at = 0;
    for (&n, &m) in gt_counts.iter().zip(&pred_counts) {
        similarity.push(
            Matrix::new(n, m, similarity_flat[at..at + n * m].to_vec()).expect("frame shape"),
        );
        at += n * m;
    }
    assert_eq!(at, similarity_flat.len(), "similarity size");
    let geometry = match as_str(&record["geometry"]) {
        "boxes" => {
            let to_boxes = |flat: Vec<f64>, counts: &[usize]| -> Vec<Vec<[f64; 4]>> {
                split_frames(&flat, counts, 4)
                    .into_iter()
                    .map(|frame| frame.as_chunks::<4>().0.to_vec())
                    .collect()
            };
            Geometry::Boxes {
                gt: to_boxes(arrays.f64s("gt_boxes"), &gt_counts),
                pred: to_boxes(arrays.f64s("pred_boxes"), &pred_counts),
            }
        }
        "masks" => Geometry::Masks {
            gt: masks(&arrays, "gt", &gt_counts),
            pred: masks(&arrays, "pred", &pred_counts),
        },
        other => panic!("unknown geometry {other}"),
    };
    let parts = SequenceDataParts {
        name: as_str(&record["name"]).to_string(),
        num_timesteps,
        num_gt_ids: as_usize(&record["num_gt_ids"]),
        num_pred_ids: as_usize(&record["num_pred_ids"]),
        num_gt_dets: as_usize(&record["num_gt_dets"]),
        num_pred_dets: as_usize(&record["num_pred_dets"]),
        gt_ids: split_frames(&arrays.usizes("gt_ids"), &gt_counts, 1),
        pred_ids: split_frames(&arrays.usizes("pred_ids"), &pred_counts, 1),
        pred_confidences: split_frames(&arrays.f64s("pred_confidences"), &pred_counts, 1),
        geometry,
        similarity,
    };
    CaseSequence {
        key: as_str(&record["key"]).to_string(),
        data: SequenceData::new(parts).expect("dumped sequence is consistent"),
    }
}

pub fn metric_by_name(name: &str) -> Box<dyn Metric> {
    use moteval::metrics::{Clear, Count, Hota, Identity, JAndF, TrackMap};
    match name {
        "HOTA" => Box::new(Hota),
        "CLEAR" => Box::new(Clear),
        "Identity" => Box::new(Identity),
        "Count" => Box::new(Count),
        "JAndF" => Box::new(JAndF),
        "TrackMAP" => Box::new(TrackMap),
        other => panic!("unknown metric {other}"),
    }
}

fn find<'a>(by_metric: &'a [(String, Scores)], metric: &str) -> Option<&'a Scores> {
    by_metric
        .iter()
        .find(|(name, _)| name == metric)
        .map(|(_, scores)| scores)
}

pub fn check_case_against_oracle(case: &Case, metric: &dyn Metric) -> usize {
    let name = metric.name();
    let label = format!("{}/{}.{name}", case.fixture, case.case);
    let mut checked = 0;
    let mut combined = Vec::new();
    let mut declared_per_sequence = Vec::new();
    for view in &case.views {
        let per_sequence: Vec<Scores> = view
            .sequences
            .iter()
            .map(|seq| {
                metric
                    .eval_sequence(&seq.data)
                    .unwrap_or_else(|err| panic!("{label}: {}: {err}", seq.key))
            })
            .collect();
        declared_per_sequence.extend(per_sequence.iter().map(|s| s.declared(metric.fields())));
        let refs: Vec<&Scores> = per_sequence.iter().collect();
        combined.push(
            metric
                .combine_sequences(&refs)
                .unwrap_or_else(|err| panic!("{label}: combine_sequences: {err}")),
        );
    }
    let mut check = |section: &str, got: &Scores| {
        if let Some(want) = case.oracle_section(section).and_then(|s| find(s, name)) {
            assert_scores_bit_equal(want, got, &format!("{label} {section}"));
            checked += want.len();
        }
    };
    match case.procedure.as_str() {
        "per_class" if case.views.len() == 1 => check("combined", &combined[0]),
        "per_class" => {
            let refs: Vec<&Scores> = combined.iter().collect();
            let class_averaged = metric
                .combine_classes_class_averaged(&refs)
                .unwrap_or_else(|err| panic!("{label}: class_averaged: {err}"));
            let det_averaged = metric
                .combine_classes_det_averaged(&refs)
                .unwrap_or_else(|err| panic!("{label}: det_averaged: {err}"));
            check("class_averaged", &class_averaged);
            check("det_averaged", &det_averaged);
        }
        "sequences_as_classes" => {
            let refs: Vec<&Scores> = declared_per_sequence.iter().collect();
            let class_averaged = metric
                .combine_classes_class_averaged(&refs)
                .unwrap_or_else(|err| panic!("{label}: class_averaged: {err}"));
            check("class_averaged", &class_averaged);
        }
        other => panic!("{label}: unknown procedure {other}"),
    }
    checked
}
