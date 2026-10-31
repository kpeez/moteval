# moteval Rust port

A Rust port of moteval that must give the same bits as Python moteval, which gives the
same bits as TrackEval `12c8791b`. The Python package stays and becomes a thin wrapper
over `moteval-py` when the port is integrated.

## Layout

```
rust/
  Cargo.toml     workspace; `cargo test` covers the default member, moteval
  moteval/       lib + src/bin/moteval.rs; zero runtime dependencies
  moteval-py/    PyO3 cdylib built by maturin (pyo3 abi3-py310, numpy)
  testdata/      inputs and expected outputs dumped from Python (see its README)
  scripts/       dump scripts (Python: standard library, numpy, scipy, moteval)
```

Modules of the `moteval` crate, each ported from the Python module of the same role:

| Module | Python | Holds |
|---|---|---|
| `model` | `data/model.py`, `data/protocol.py` (`Protocol`) | `SequenceData`, `GtSequence`, `MaskGtSequence`, `MOTDataset`, `FrameConvention`, `Protocol`, `Matrix` |
| `numeric` | numpy, builtins | every reduction order, `arange`, `linspace`, rounding, `maximum`/`py_max` |
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
MOTEVAL_RUST_TESTDATA=artifacts/rust-port/testdata cargo test --manifest-path rust/Cargo.toml
```

Regenerate the test data with the reference interpreter:

```sh
uv run python rust/scripts/dump_numeric.py     # numeric.json, box_similarity.json
uv run python rust/scripts/dump_lsap.py        # lsap/
uv run python rust/scripts/dump_scenarios.py   # scenarios/ (synthetic, committed)
MOTEVAL_DATA_ROOT=/data/tracking/external \
  uv run python rust/scripts/dump_scenarios.py --real-data --jobs 8   # artifacts/rust-port/testdata
uv run python rust/scripts/gen_printable.py    # moteval/src/pyfmt/printable.rs
uv run python rust/scripts/dump_pyfmt.py       # pyfmt.json
uv run python rust/scripts/dump_results.py     # exports.json
uv run python rust/scripts/dump_cli.py         # cli.json
```

`moteval-py` builds the extension `moteval._moteval` (`pyproject.toml` uses maturin):
`mise run install` (or any `uv run`) rebuilds it after a Rust change, and
`tests/test_bindings.py` checks it against the Python code. Build a wheel with
`uvx maturin build --release --locked`.

## Rules for every package

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
- No `unsafe`, no `rayon`, no SIMD, no `mul_add`.

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
