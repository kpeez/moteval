# rust/testdata

Inputs and expected outputs dumped from Python for the Rust tests. Never hand-edit these
files: regenerate them with the scripts in `rust/scripts/`, run with the reference
interpreter (`uv run python ...`: CPython 3.14.8, numpy 2.5.1, scipy 1.18.0,
pycocotools 2.0.11).

Every float is stored as its IEEE-754 `u64` bit pattern (a JSON integer), because JSON
numbers cannot carry NaN or exact bits. Tests compare `f64::to_bits`; any NaN equals any
NaN.

| Path | Script | Read by |
|---|---|---|
| `numeric.json` | `dump_numeric.py` | `tests/numeric.rs` |
| `box_similarity.json` | `dump_numeric.py` | `tests/box_similarity.rs` |
| `lsap/<category>.json`, `lsap/special.json` | `dump_lsap.py` | `tests/lsap.rs` |
| `scenarios/<fixture>/<case>.json` | `dump_scenarios.py` | `tests/harness_cases.rs` and the metric tests |

## Seeded inputs

`numeric.json`, `box_similarity.json`, and `lsap/` store seeds, not inputs.
`rust/scripts/testgen.py` and `rust/moteval/tests/harness/testgen.rs` generate the same
inputs from a seed, draw for draw (splitmix64; every draw is exact). Each case stores
`inputs_fnv` (or `fnv`): FNV-1a 64 over the little-endian bytes of the generated floats.
A generator mismatch fails as "generator drift", not as a wrong answer.

## Scenario cases

One case is one fixture entry: its `SequenceData` inputs, the oracle numbers, and Python
moteval's own scores. `dump_scenarios.py` checks that Python moteval equals the oracle bit
for bit before it writes a case (`oracle_fields_checked` counts the fields).

Synthetic cases (`tests/fixtures/synthetic_{box,mots,trackmap}.json`) are committed here,
one self-contained JSON file each. Real-data cases (`real_data.json`: DanceTrack val,
SportsMOT val, one MOTS20 sequence; `reference_benchmarks.json`: 20 splits of the nine
local benchmarks) are too large to commit. `dump_scenarios.py --real-data` writes them to
`artifacts/rust-port/testdata/<fixture>/<case>/`, and the Rust tests find them through
`MOTEVAL_RUST_TESTDATA`. When it is unset or a case is absent, a test prints
`SKIPPING REAL-DATA GATE: ...` and returns; a skipped gate never counts as passed.

### `case.json`

```json
{
  "format": 1,
  "fixture": "synthetic_box",
  "case": "misses",
  "procedure": "per_class",
  "protocol": {"name": "parity", "first_frame": 1, "eval_classes": [1], ...},
  "metrics": ["HOTA", "CLEAR", "Identity", "Count"],
  "views": [{"class": 1, "sequences": [<sequence>, ...]}],
  "oracle": {"combined": [[metric, [[field, value], ...]], ...]},
  "moteval": {
    "views": [{"class": 1, "per_sequence": [[key, [[metric, scores]]]], "combined": [[metric, scores]]}],
    "class_averaged": [[metric, scores]],
    "det_averaged": [[metric, scores]]
  },
  "oracle_fields_checked": 47
}
```

- `views` holds one entry per evaluated class, in `eval_classes` order, each with that
  class's sequences in dataset order: exactly what `moteval.evaluate` scores. TrackMAP's
  spec-built scenarios have one view with `"class": null`.
- `procedure` says how the oracle numbers come from the views:
  - `per_class`: score each sequence, then `combine_sequences` per view. One view: compare
    with `oracle.combined`. Several views: run `combine_classes_class_averaged` and
    `combine_classes_det_averaged` over the views' raw combined state, in class order, and
    compare with `oracle.class_averaged` and `oracle.det_averaged`.
  - `sequences_as_classes`: one view; its sequences' declared per-sequence scores go to
    `combine_classes_class_averaged` (`synthetic_box/combine_classes_class_averaged`).
- A score value is `{"shape": [...], "bits": [...]}`: shape `[]` is a scalar, `[n]` an
  array, `[rows, cols]` a C-order matrix. Scores are `[[field, value], ...]` in Python's
  dict order, which is the export order.
- `oracle` holds only the fields the fixture freezes; TrackMAP's det-averaged value (the
  one allowed divergence) is not frozen, so the multi-class box case has no TrackMAP in
  `oracle.det_averaged`.
- `moteval` holds Python moteval's results: per-sequence scores are raw (private fields
  included) for synthetic cases and declared fields only for real data; `combined`,
  `class_averaged`, and `det_averaged` are raw.

### Sequences

A sequence record holds the scalars of `SequenceData` (`key`, `name`, `num_timesteps`,
`num_gt_ids`, `num_pred_ids`, `num_gt_dets`, `num_pred_dets`, `geometry`) and its
frame-major fields as flat arrays. `key` is the sequence's key in Python's results (the
dataset sequence name, or the TrackMAP spec key).

| Array | dtype | Length | Content |
|---|---|---|---|
| `gt_counts`, `pred_counts` | i64 | `num_timesteps` | detections per frame |
| `gt_ids`, `pred_ids` | i64 | sum of counts | dense ids, frame after frame |
| `pred_confidences` | f64 | sum of `pred_counts` | |
| `similarity` | f64 | sum of `gt_t * pred_t` | each frame's `gt x pred` matrix, C order |
| `gt_boxes`, `pred_boxes` | f64 | 4 x sum of counts | `xywh` (box sequences) |
| `{gt,pred}_mask_heights`, `_widths`, `_count_lens` | i64 | sum of counts | mask sequences |
| `{gt,pred}_mask_counts` | u8 | sum of `_count_lens` | compressed RLE strings, concatenated |

Synthetic records hold each array inline: `{"dtype": "f64", "data": [bits]}`,
`{"dtype": "i64", "data": [ints]}`, or `{"dtype": "ascii", "data": "..."}` for mask counts.
Real-data cases reference one header per sequence instead (`{"header": "v0_s000.json"}`).
The header is the same record with `"bin": "v0_s000.bin"`, and each array is
`{"dtype": "f64" | "i64" | "u8", "offset": <bytes>, "len": <elements>}` into that file:
flat little-endian values, each array starting 8-byte aligned. Rust reads them with
`f64::from_le_bytes` and `i64::from_le_bytes`.
