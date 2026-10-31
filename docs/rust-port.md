# The Rust port

moteval's scoring is Rust (`rust/`). The Python package is a thin wrapper over the
extension `moteval._moteval` and keeps every public name of `moteval/__init__.py`. The
port's acceptance rule: every number equals, bit for bit, what the pure-Python moteval of
the `py-reference` tag gives, and that moteval equals TrackEval `12c8791b` (except the
TrackMAP det-averaged fix below). This page records the gates, the benchmark, and every
difference from the pure-Python package.

## How parity is checked

- **Oracle fixtures** (`tests/fixtures/{synthetic_box,synthetic_mots,synthetic_trackmap,real_data}.json`,
  TrackEval numbers): the Python suite (`tests/test_parity.py`, `tests/test_parity_real.py`)
  now runs them through the extension, and the Rust tests
  (`clear_identity_count.rs`, `hota_trackmap.rs`, `jf.rs`) run each metric on the dumped
  `SequenceData` of every fixture entry.
- **Python moteval's own scores**: `rust/scripts/dump_*.py` take expected values from the
  pure-Python reference (the `py-reference` tag in `artifacts/rust-port/ref-venv`) and
  store them as float bits in `rust/testdata/` (committed) and
  `artifacts/rust-port/testdata/` (real data). The Rust tests compare every field, private
  combiner state and dict order included, with `f64::to_bits` (a NaN equals a NaN).
- **The reference fixture** (`tests/fixtures/reference_benchmarks.json`: combined results on
  all nine local benchmarks, every split with ground truth, seeded predictions):
  `tests/test_reference_real.py` checks it through Python, and
  `rust/moteval/tests/reference_end_to_end.rs` through the Rust pipeline alone (Rust
  loaders, `eval::evaluate`), with every per-sequence score against Python as well.
- **Reproducibility**: run with the reference venv, every dump script, the loader-case
  recorder, and `scripts/regen_parity_fixtures.py` (TrackEval) rewrite their committed
  files byte for byte.
- **Entry points**: the reference CLI, the Rust-backed Python CLI, and the Rust binary
  write byte-identical `--out-json` files on the four benchmark cases below.

What the port copies on purpose (details in `rust/README.md` and the module docs): every
reduction order (`numeric`: numpy pairwise sums, row-wise means, plain left folds, cumsum),
scipy 1.18.0's `linear_sum_assignment` line by line, pycocotools 2.0.11's mask API, CPython
3.14.8's `set` iteration order for TrackMAP's frame union, numpy's `uint8` wrap for the J&F
decay bins, `round_ties_even` for `np.round`, numpy's `log10` for CLEAR's MOTAL (Intel
SVML's kernel on an AVX-512 Linux host, libm elsewhere), and CPython's text formats (`repr(float)`,
`format(x, "1.5g")`, `json.dumps`, `csv.writer`, argparse 3.14) for the CLI and exports.

## Gates

All on the final commit of the integration branch (the PR body lists the exact commands
and outputs).

| Gate | Result |
| --- | --- |
| `cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings` | clean |
| `cargo test` (committed test data) | every binary passes; real-data tests print `SKIPPING REAL-DATA GATE` |
| `cargo test` with `MOTEVAL_RUST_TESTDATA` and `MOTEVAL_DATA_ROOT` | every binary passes, zero skips: 23 real-data cases, 20 loader splits, 20 end-to-end cases (1,214 combined fields equal the reference, 1,150 sequences equal Python) |
| `mise run check` | ruff and pyrefly clean |
| `mise run test` | passes; one skip, the opt-in network test |
| `mise run test-real` (`MOTEVAL_DATA_ROOT=/data/tracking/external`) | 38 passed, zero skips |
| Committed test data and parity fixtures regenerate identically | no change under `rust/testdata/` or `tests/fixtures/` |
| CI | `check`, `test` (Python 3.10 to 3.14), `rust`, and both wheel targets (Linux x86_64, macOS arm64) |

## Performance

`rust/scripts/bench_cli.py` on this host (AVX-512 x86-64, single thread): seeded predictions
from `dump_loaders.py`, every applicable metric (boxes: HOTA, CLEAR, Identity, Count,
TrackMAP; MOTS20: HOTA, CLEAR, Identity, Count, J&F). Wall time is the hyperfine mean of 10
runs after 3 warm-ups; peak RSS is the largest of three `/usr/bin/time -v` runs. Raw output:
`artifacts/rust-port/bench/`.

| Case | Reference CLI | Python CLI (Rust) | Rust binary | Speed-up (binary) | Peak RSS: reference / Python / binary |
| --- | --- | --- | --- | --- | --- |
| DanceTrack val (25 seqs) | 4.84 s | 0.49 s | 0.39 s | 12.3x | 257 / 72 / 41 MiB |
| SportsMOT val (45 seqs) | 5.78 s | 0.58 s | 0.48 s | 12.1x | 278 / 78 / 48 MiB |
| MOTS20 train (4 seqs) | 38.59 s | 4.18 s | 3.82 s | 10.1x | 233 / 174 / 143 MiB |
| ChimpACT val (17 seqs) | 2.19 s | 0.20 s | 0.11 s | 19.6x | 126 / 47 / 16 MiB |

The Python CLI is the console script, which runs the same Rust CLI after the interpreter
and numpy start (about 0.1 s). The real-data Python gate (`mise run test-real`) went from
about 260 s to about 80 s.

## Differences from the pure-Python package

Numbers: none. The one divergence from TrackEval is inherited: TrackMAP's
`combine_classes_det_averaged` weights classes by detections, where upstream's is a copy
of its class-averaged combiner.

Behavior (all non-numeric):

- `evaluate` runs the six built-in metrics only; a user-defined `Metric` subclass raises
  `ValueError`. Each metric object's `eval_sequence` and combiners still work on their own.
- `evaluate` on a dataset with no sequences raises `ValueError` ("dataset 'x' has no
  sequences to score") before reading any file. Python gave integer zeros (CLEAR,
  Identity, Count), an `IndexError` (HOTA), or numpy's "need at least one array to
  concatenate" (TrackMAP). Called directly with no results, a combiner raises
  `ValueError` ("HOTA: no results to combine"), and called with a score dict that lacks
  a field the combiner reads, it raises `pyo3_runtime.PanicException`, which derives
  from `BaseException`, not `Exception` (Python raised `KeyError`).
- Scores are Python `float` or float64 numpy arrays. Some CLEAR fields were `np.float64`
  and the empty-dataset zeros were `int`; the values are the same.
- Exceptions keep their class and text, with three exceptions: a file error is `OSError`
  (not its subclass, such as `FileNotFoundError`); a decode error is `ValueError` with
  `UnicodeDecodeError`'s text (its base class); a `json.JSONDecodeError` has `doc=None`.
- Inputs Python accepts but Rust cannot hold are `ValueError`: integers beyond 64 bits in
  MOT and MOTS rows, negative or oversized sequence lengths and mask sizes, JSON ids that
  are not integers, numbers beyond `i128`, lone surrogates, JSON nested deeper than 2,000
  levels, and file names that are not UTF-8. A mask counts string whose runs do not cover
  the mask is an error (pycocotools reads past its buffer there).
- CLI: help and usage text use argparse 3.14's layout for an 80-column terminal, without
  color, on every Python version. Where Python died with a traceback (an I/O error), the
  CLI prints `moteval: error: <message>` and exits 1; `moteval.cli.main` raises
  `SystemExit(1)` there, and returns 0 (not `SystemExit(0)`) after `-h`.
- `scipy` and `pycocotools` are no longer runtime dependencies.

## Open risks

- **CLEAR MOTAL's `np.log10` depends on the host, in numpy and in the port alike.** numpy
  2.5.1's Linux x86-64 wheels compute float64 `log10` with Intel SVML
  (`__svml_log108_ha`) when the CPU has AVX-512 F, CD, BW, DQ, and VL, and with libm
  otherwise. The two differ in the last bit on 144,516 of the integers 1 to 10^7, and
  MOTAL (`(TP - FP - log10(IDSW)) / max(1, TP + FN)`) shows it when `|TP - FP|` is small
  (the parity review found IDSW 11 and 40). `numeric::log10` makes the same choice at run
  time: `numeric::log10_svml`, a scalar port of the SVML kernel, or `f64::log10`. The
  port equals numpy's SVML path on 12,000,007 inputs (the integers 1 to 10^7, two
  million random positive doubles, and seven edge values) and on 448 inputs where an exact `1/m` in place of
  `vrcp14pd` would change the result. `vrcp14pd` is Intel's reference `RCP14D`, which
  equals this AMD Zen 5 host's instruction on every 16-bit mantissa prefix; a CPU whose
  `vrcp14pd` differs would change numpy's results there, not the port's. numpy's
  `NPY_DISABLE_CPU_FEATURES` can move numpy off SVML; the port does not read it.
- **TrackMAP's frame-union order replays CPython 3.14.8's `set` hash table** (pinned by a
  4,304-case fuzz test). A different reference interpreter needs a new dump.
- **numpy's sum order follows strides.** Every similarity matrix Python moteval built was
  C-contiguous, and the Rust `Matrix` is C order. A `SequenceData` built by hand with a
  Fortran-order matrix is converted to C order at the boundary, so Rust sums it in C
  order where the pure-Python metric would have followed the strides (a possible
  last-bit difference on such input only).
- **The J&F `uint8` cast** emulates x86-64 `cvttsd2si` (wrap mod 256 through `i32`); values
  of 2^31 frames or more could differ on other architectures.

## Tests moved to Rust

Python tests whose subject is now Rust code were removed; these Rust tests cover them:

| Removed Python test | Rust coverage |
| --- | --- |
| `tests/test_bindings.py` (Rust CLI, writers, and errors against Python) | `tests/cli.rs` (103 argument lists, stdout, stderr, exit code), `tests/exports.rs` (14 results byte for byte, 11 pre-scoring errors) |
| `test_data.py`: id densification, class views, ignore regions, the preprocessing engine, box IoU/IoA | `tests/loader_cases.rs` replays every call of the `py-reference` `test_data.py` and `test_loaders.py` (194 results, 25 errors) |
| `test_loaders.py`: `MOTChallengeConfig.class_id` stamping | `tests/loader_cases.rs` (`load_layout` cases) |
| `test_masks.py`: mask IoU and IoA | `tests/rle.rs` (1,000 masks against pycocotools), `tests/jf.rs` (similarity of every MOTS frame) |

Tests that used internal functions (`build_sequence_data`, the loader modules) now use the
public API: `BENCHMARKS`, `load_motchallenge`, and `evaluate`.
