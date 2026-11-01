# moteval Rust port

The Rust implementation of moteval. It gives the same bits as the pure-Python moteval
of the `py-reference` tag, which gives the same bits as TrackEval `12c8791b`. The Python
package (`moteval/`) is a thin wrapper over `moteval-py`: it keeps the public data types
and calls Rust for loading, reading, scoring, the exports, and the CLI.
`docs/rust-port.md` has the gate results, the benchmark, and the differences from the
pure-Python package.

## Layout

```
rust/
  Cargo.toml     workspace; `cargo test` covers the default member, moteval
  moteval/       lib + src/bin/moteval.rs; zero runtime dependencies
  moteval-py/    PyO3 cdylib built by maturin (pyo3 abi3-py310, numpy)
  testdata/      inputs and expected outputs dumped from Python (see its README)
  scripts/       dump scripts (Python, run with the reference venv: see below)
```

Modules of the `moteval` crate, each ported from the Python module of the same role:

| Module | Python | Holds |
|---|---|---|
| `model` | `data/model.py`, `data/protocol.py` (`Protocol`) | `SequenceData`, `GtSequence`, `MaskGtSequence`, `MOTDataset`, `FrameConvention`, `Protocol`, `Matrix` |
| `numeric` | numpy, builtins | every reduction order, `arange`, `linspace`, rounding, `maximum`/`py_max`, `log10` (numpy's SVML or libm choice) |
| `lsap` | scipy `linear_sum_assignment` | the scipy 1.18.0 solver, line by line |
| `similarity` | `data/similarity.py` | box IoU and IoA; mask IoU and IoA |
| `rle` | pycocotools | `RleMask` and the mask API |
| `metrics` | `metrics/` | the `Metric` trait, `Scores`, `Value`, the shared combine helpers, one module per metric |
| `formats`, `convert`, `protocol`, `benchmarks` | `formats.py`, `data/convert.py`, `data/protocol.py`, `benchmarks/` | readers, conversion, the preprocessing engine, loaders |
| `eval`, `results`, `pyfmt`, `cli` | `eval.py`, `results.py`, `cli.py` | orchestration, results and exports, Python-compatible text, the CLI |

## Commands

From the repository root (Rust comes from `mise.toml`):

```sh
cargo fmt --manifest-path rust/Cargo.toml --all --check
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
MOTEVAL_RUST_TESTDATA=artifacts/rust-port/testdata MOTEVAL_DATA_ROOT=/data/tracking/external \
  cargo test --manifest-path rust/Cargo.toml
```

The real-data loader gate (`tests/loaders_real.rs`) also loads the benchmarks themselves,
so it needs `MOTEVAL_DATA_ROOT` as well as `MOTEVAL_RUST_TESTDATA`.

### The reference venv

Expected values come from the pure-Python moteval, not from this checkout's package
(which is Rust): the `py-reference` tag, installed into its own venv. Never install it
into the project `.venv`; both packages are named `moteval`.

```sh
uv venv --python 3.14.8 artifacts/rust-port/ref-venv
uv pip install --python artifacts/rust-port/ref-venv/bin/python \
  "moteval @ git+file://$PWD@py-reference" numpy==2.5.1 scipy==1.18.0 \
  pycocotools==2.0.11 pytest
```

Every dump script that reads moteval puts the repository root at the end of `sys.path`
and calls `rust/scripts/reference.py`'s `require_reference()`, which exits unless
`moteval` is the reference. `record_loader_cases.py` records the `py-reference` tag's
own `tests/test_loaders.py` and `tests/test_data.py` (extracted with `git archive`).

Regenerate the test data (`REF=artifacts/rust-port/ref-venv/bin/python`); every committed
file comes back byte-identical:

```sh
$REF rust/scripts/dump_numeric.py     # numeric.json, box_similarity.json
$REF rust/scripts/dump_lsap.py        # lsap/
$REF rust/scripts/dump_set_order.py   # set_order.json (TrackMAP frame-union order)
$REF rust/scripts/dump_metric_internals.py  # hota_trackmap_internals.json
$REF rust/scripts/dump_masks.py       # rle*.json, jf_boundary.json, jf_sequences.json
$REF rust/scripts/dump_scenarios.py   # scenarios/ (synthetic, committed)
$REF rust/scripts/dump_clear_identity_count.py   # fuzz/clear_identity_count/
$REF rust/scripts/dump_pytext.py         # pytext/ (CPython text verdicts)
$REF rust/scripts/record_loader_cases.py # loader_cases/ (py-reference's test_loaders.py, test_data.py)
$REF rust/scripts/dump_pyfmt.py       # pyfmt.json
$REF rust/scripts/dump_results.py     # exports.json
$REF rust/scripts/dump_cli.py         # cli.json
MOTEVAL_DATA_ROOT=/data/tracking/external \
  $REF rust/scripts/dump_loaders.py --jobs 8   # artifacts/rust-port/testdata/loaders_real
MOTEVAL_DATA_ROOT=/data/tracking/external \
  $REF rust/scripts/dump_scenarios.py --real-data --jobs 8   # artifacts/rust-port/testdata
uv run python rust/scripts/gen_printable.py    # moteval/src/pyfmt/printable.rs
uv run python rust/scripts/gen_pytext_tables.py   # moteval/src/formats/pytables.rs
```

`dump_loaders.py` writes the prediction files `dump_scenarios.py --real-data` scores, so
regenerate both together. `tests/reference_end_to_end.rs` reads both: it runs the whole
Rust pipeline (`load_dataset`, then `evaluate` on `loaders_real/<case>/pred`) on all
nine benchmarks and checks every combined field against
`tests/fixtures/reference_benchmarks.json` and every per-sequence score against Python.

`rust/scripts/bench_cli.py` times the reference CLI, the Rust-backed Python CLI, and the
Rust binary on four benchmarks, after checking that their JSON outputs are byte-identical
(`docs/rust-port.md` has the numbers).

`moteval-py` builds the extension `moteval._moteval` (`pyproject.toml` uses maturin):
`mise run install` (or any `uv run`) rebuilds it (release profile) after a Rust change,
and the Python suite (`mise run test`, `mise run test-real`) runs through it. Build a
wheel with `uvx maturin build --release --locked`.

## Rules

- Bit-identical results. Compare `f64::to_bits` (any NaN equals any NaN); never a
  tolerance. Load the `moteval-float-parity` skill before writing numeric code.
- Copy each Python call site's reduction order through `moteval::numeric`; its module
  documentation maps every numpy and builtin call to its function. `np.maximum` propagates
  NaN and Python's `max` does not: use `numeric::maximum` or `numeric::py_max` to match
  the Python line.
- `Scores` is insertion-ordered and its order is the export order: insert fields in the
  order the Python dict was built. Fields that start with `_` are private combiner state.
- Combiners take results in Python's dict order: sequences in dataset order, classes in
  `eval_classes` order.
- Errors are a plain `enum` per module with a hand-written `Display` whose text equals
  the Python message. Quote names with `pyfmt::repr_str` where Python uses `{name!r}`.
- No `unsafe`, no `rayon`, no SIMD, no `mul_add`. One exception: `numeric::log10_svml`
  ports numpy's SVML `log10` kernel, whose `vfmadd`/`vfmsub` instructions round once, so
  each one is a `mul_add` there.

## Test harness

`moteval/tests/harness/` is shared by the integration tests (`mod harness;`):

- `load_synthetic_case(fixture, case)` and `load_real_case(root, fixture, case)` rebuild a
  dumped case's `SequenceData` views, oracle, and Python moteval scores.
- `check_case_against_oracle(&case, &metric)` runs one metric end to end, as `evaluate`
  does, and asserts every oracle field it covers.
- `assert_scores_bit_equal(expected, got, label)` compares every expected field bit for
  bit; `assert_field_order` checks Python's dict order.
- `real_testdata_dir(what)` reads `MOTEVAL_RUST_TESTDATA`, or prints the
  `SKIPPING REAL-DATA GATE` line and returns `None`.
