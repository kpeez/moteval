# moteval — agent guide

Issue tracker: Linear. When Linear's free issue limit blocks a new issue, file it in
GitHub Issues on this repo instead and close it with `Fixes #N`.

MOT evaluation library: a from-scratch TrackEval rewrite that must produce **bit-identical
numbers** to official TrackEval commit `12c8791b`. Evaluation only — it never runs models.
The implementation is Rust (`src/`); the Python package `moteval/` is a thin wrapper that
keeps the public names.

## Non-negotiables

- Replicate upstream's *intentional* quirks exactly: eps-guarded thresholds
  (`similarity >= alpha - eps` in HOTA, `< threshold - eps` in CLEAR), CLEAR's `1000×`
  same-ID cost bonus, Identity's `(G+T)×(G+T)` block cost matrix with `1e10` off-diagonals,
  TrackMAP's right-to-left monotonic precision + `np.searchsorted`, the box-path Hungarian
  below-threshold fill of `0` vs the MOTS-path fill of `-10000` (`Protocol.matching_fill`),
  and J&F's uint8 decay bins wrapping past 255 frames.
- Sole permitted numeric divergence: TrackMAP `combine_classes_det_averaged` (upstream bug;
  we implement the correct detection-weighted average). Every other metric's class
  combiners, det-averaged included, match upstream and are oracle-checked.
- Fix non-numeric hazards freely: ID densification uses dicts, never `np.max(ids)+1` arrays.
- Masks are pycocotools RLE; encode from Fortran-contiguous `(h, w, n)` arrays only.
- Everything converges to `MOTDataset` → frozen frame-major `SequenceData`; metrics consume
  `SequenceData` alone (precomputed per-frame similarity; TrackMAP also reads boxes and
  J&F masks, each declared in `Metric.geometry`).
- Frame-indexing is a declared loader parameter; out-of-range frames raise — a silent drop
  is never acceptable. Regression tests must number predictions independently of GT.
- Per-benchmark preprocessing is a declarative `Protocol` executed by the shared engine —
  never subclass-hook preprocessing logic.
- Floats are compared with `f64::to_bits`, never a tolerance. Every reduction goes
  through `numeric`, which holds the upstream order. Use no rayon, SIMD, or `mul_add`;
  `numeric::log10_svml` is the one exception, because SVML's fused instructions round once.
- `Scores` is insertion-ordered, and that order is the export order. Fields that start
  with `_` are private combiner state.
- Each module's error enum has a hand-written `Display`. Tests pin that text, so change it
  only on purpose.

## Layout

- `Cargo.toml`, `src/` — one crate, `moteval`: the library and the `moteval` binary
  (`src/main.rs`), with zero runtime dependencies. The `python` feature adds pyo3 and
  numpy and compiles `src/python.rs`, the extension `moteval._moteval` (stubs in
  `moteval/_moteval.pyi`); maturin builds with that feature (`[tool.maturin]`).
  `numeric` holds every numpy and builtin reduction order, `lsap` is scipy's solver,
  `rle` is pycocotools' mask API, and `pyfmt` writes CPython-compatible text.
- `moteval/` — the Python package: the frozen dataclasses (`data/model.py`,
  `data/protocol.py` `Protocol`, `formats.py` `Track`/`MaskTrack`, `results.py`) and thin
  functions over the extension: `evaluate` (`eval.py`), the loaders
  (`benchmarks/__init__.py`: `BENCHMARKS`, `load_dataset`, `load_motchallenge`,
  `load_mots`, `default_data_root`), the readers and writers, the exports
  (`results.to_json`/`to_csv`/`to_json_dict`), the metric classes (`metrics/`:
  `RustMetric` subclasses whose `eval_sequence` and combiners run in Rust), and
  `cli.main`. Rust errors cross as `RustError(class, message)`; `moteval/_rust.py`
  re-raises the Python class. `evaluate` takes the built-in metrics only.
- `evaluate` scores each class in `protocol.eval_classes`. A single-class run fills
  `EvaluationResult.per_sequence`/`combined`; a multi-class run fills `per_class`,
  `class_averaged` and `det_averaged` instead (the class combiners take the raw per-class
  state, because TrackMAP weights by private `_num_dt_*` fields). Box predictions give
  their class in column 8, read (`read_mot(class_column=True)`) only for multi-class
  protocols. Single-class JSON, CSV and table output must not change. Every sequence
  needs a `<seq>.txt` prediction file (an empty file means no predictions); `evaluate`
  raises `ValueError` naming every missing sequence before it scores any, as TrackEval
  does. Tests that mean "no predictions" write an empty file. `evaluate` also raises
  `ValueError` before scoring when a metric's `Metric.geometry` (boxes or masks) does not
  match a sequence.
- The `moteval` console script is the Rust CLI (`moteval._moteval:main`), the same code
  as the `moteval` binary; `moteval.cli.main(argv)` runs it in-process.
- `scripts/download_benchmarks.py` — dev-only benchmark downloader (`list/status/download`)
- `scripts/reference_results.py` — scores the nine benchmarks with seeded predictions; writes
  per-sequence and combined JSON and the reference fixture
- `tests/` — flat Python suite (test_metrics.py, test_parity.py, test_parity_real.py,
  test_data.py, test_loaders.py, test_loaders_real.py, test_reference_real.py,
  test_masks.py, test_cli.py, test_download.py) that runs through the extension,
  `tests/fixtures/*.json`
  (frozen TrackEval oracle numbers, except `reference_benchmarks.json`), `tests/scenarios.py` (shared scenario definitions),
  `tests/perturb.py` (seeded perturbed predictions), `tests/rle.py` (RLE encoding for mask
  fixtures), `tests/conftest.py` (in-memory toy dataset and a two-class dataset, with
  `write_perfect_predictions`, `write_flawed_predictions` and
  `write_two_class_predictions` to write their prediction files, and `write_toy_layout`
  to write the toy dataset to disk for CLI tests). The shared helpers use the public API
  only, because the dump scripts import them under the reference venv. `tests/temp/` is
  gitignored scratch for data, not tests (see Gotchas).
- `tests/integration/` — the Rust tests: one test target (`main.rs`), the shared
  `harness/`, and the committed expected values in `tests/integration/data/`.
- `scripts/testdata/` — the scripts that write `tests/integration/data/` (see Gotchas).
  `scripts/gen_printable.py` and `scripts/gen_pytext_tables.py` regenerate
  `src/pyfmt/printable.rs` and `src/formats/pytables.rs` from the interpreter's Unicode
  database.
- CI: `ci` runs `check`, the Python suite on 3.10 to 3.14 (every Python job installs
  Rust, because `uv sync` builds the extension), and `rust` (the cargo commands below,
  without real data); `wheels` builds abi3 wheels for Linux x86_64 and macOS arm64.

## Commands

```sh
mise run install   # uv sync --locked, prek hooks (main checkout only)
mise run check     # ruff format, ruff check --fix, pyrefly check
mise run test      # pytest (testpaths: tests/; real-data gate deselected by default)
mise run test-real # slow real-data gate: parity + raw-GT loader checks (needs the data root)
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings   # also checks src/python.rs
cargo test   # real-data tests skip loudly without the env:
MOTEVAL_REAL_TESTDATA=artifacts/rust-port/testdata MOTEVAL_DATA_ROOT=<root> cargo test
```

`check`, `test`, and `test-real` must pass before any PR, and so must the cargo commands
when the PR touches Rust.

`mise.toml` holds the tasks only. `rust-toolchain.toml` pins Rust, and uv pins Python
through `.python-version`. Run `mise run install` first in every fresh clone or git
worktree: the venv is an editable install, and it rebuilds the Rust extension when a
Rust source changes (`[tool.uv] cache-keys`). Host-only settings,
such as `MOTEVAL_DATA_ROOT`, go in the gitignored `mise.local.toml`. `mise run` trusts its own
config; run `mise trust` only to activate that env in your shell. `mise run install`
installs the git hooks only in the main checkout, because a worktree shares `.git/hooks`.

## Gotchas

- Parity fixtures (`tests/fixtures/*.json`) are never hand-edited — regenerate with
  `scripts/regen_parity_fixtures.py`, which clones TrackEval @ `12c8791b`, applies
  numpy>=2 alias patches, and rewrites the JSONs. It runs under the reference venv
  (`artifacts/rust-port/ref-venv/bin/python`, with opencv-python-headless and
  scikit-image added), not the project venv: it feeds moteval's own per-sequence HOTA to
  the oracle, and that moteval must be the pure-Python reference. The oracle datasets score one class,
  so the class-combination entries score one single-class view per class
  (`tests.scenarios.class_views`). The box entry treats a scenario's sequences as
  classes. The mask entry splits one sequence by track id (`MOTS_MULTI_CLASS_TRACKS`).
- `tests/fixtures/reference_benchmarks.json` is not an oracle fixture. It freezes moteval's
  own combined results (every applicable metric, floats exact) on all nine local
  benchmarks and each split with ground truth, scored against seeded `tests/perturb.py`
  predictions. The Rust pipeline is checked against it, so it changes only when a metric
  changes on purpose. Never hand-edit it. Regenerate with
  `MOTEVAL_DATA_ROOT=<root> uv run python scripts/reference_results.py --out-dir artifacts/reference --reference-fixture tests/fixtures/reference_benchmarks.json`;
  `tests/test_reference_real.py` (`real_data`, bit for bit, `np.array_equal` with
  `equal_nan`) checks it through Python, `tests/integration/reference_end_to_end.rs`
  through the Rust pipeline alone, and both skip loudly when data is absent.
- `tests/integration/data/` is never hand-edited. Its expected values come from the
  pure-Python moteval of the `py-reference` tag, and the Rust tests compare them bit for
  bit (each float is stored as its `u64` bit pattern; any NaN equals any NaN). Each
  `scripts/testdata/dump_*.py` script, and `record_loader_cases.py`, rewrites its own
  files there, byte-identical when nothing changed. Run them, and
  `scripts/regen_parity_fixtures.py`, with the reference venv, never the project `.venv`:
  both packages are named `moteval`, and the project one would test Rust against itself.
  Each script appends the repository root to `sys.path` and exits unless `moteval` is the
  reference (`scripts/testdata/reference.py`). Build the venv once:
  `uv venv --python 3.14.8 artifacts/rust-port/ref-venv`, then
  `uv pip install --python artifacts/rust-port/ref-venv/bin/python "moteval @ git+file://$PWD@py-reference" numpy==2.5.1 scipy==1.18.0 pycocotools==2.0.11 pytest`.
  The real-data dumps go to `artifacts/rust-port/testdata` (`MOTEVAL_REAL_TESTDATA`):
  `dump_loaders.py` writes the predictions that `dump_scenarios.py --real-data` scores,
  so run both together, with `MOTEVAL_DATA_ROOT` set.
- `tests/integration/data/loader_cases/` records the `py-reference` tag's
  `test_loaders.py` and `test_data.py`, not this checkout's (`record_loader_cases.py`
  extracts them with `git archive`).
- The Rust CLI cannot see a dataset built or registered in Python. CLI tests write the
  toy dataset to disk (`write_toy_layout`) and pass `--gt <root> --split val`.
- numpy's float64 `log10` (CLEAR's MOTAL) is Intel SVML on an AVX-512 Linux host and
  libm elsewhere, and the two differ in the last bit. `numeric::log10` makes numpy's
  choice at run time; `tests/integration/data/numeric.json` holds SVML's output, so dump it on an
  AVX-512 Linux host (`dump_numeric.py` refuses otherwise).
- `mise run install` (and any `uv run`) rebuilds the extension in release mode after a
  Rust source change; a build error there is a Rust compile error.
- `data/benchmarks` is a symlink to external storage holding one dir per dataset.
  `moteval.benchmarks.default_data_root()` owns the data-root rule: `MOTEVAL_DATA_ROOT`
  if set (an empty value raises), else `data/benchmarks` relative to the working
  directory. `load_dataset(name)` reads `<data root>/<name>`, `uv run
  scripts/download_benchmarks.py download <name>` writes there (`--root` overrides),
  and `mise run test-real` reads from it. Individual loaders take a required `root` and hold
  no default path. Real-data tests skip loudly when their dataset is absent.
- GMOT-40 and ChimpACT are natively 0-indexed. Both loaders keep raw 0-indexed frame
  numbers and declare `FrameConvention(first_frame=0)` rather than shifting.
- BFT, AnimalTrack, and GMOT-40 have no `seqinfo.ini` source, so their loaders derive
  `num_timesteps` from the last annotated frame instead. This undercounts a sequence with
  no GT in its final frames — harmless for metrics but predictions past the last annotated
  frame raise the frame-out-of-range error.
- UAVDT ignore regions (`<seq>_gt_ignore.txt`) must be honored.
- Never spawn `uv run …` from inside a test that is itself running under `uv run pytest`:
  the nested invocation deadlocks on uv's project lock. CLI tests run in-process via
  `moteval.cli.main(argv)`; if a subprocess is ever unavoidable, call the installed
  `.venv/bin/moteval` entry point directly.
- pytest collects `test_*.py` files under the gitignored `tests/temp/`, because it sits
  inside `testpaths`. Keep scratch tests outside `tests/`.
- A same-size source swap within the same second can leave a stale `.pyc` that Python
  still trusts. Run mutation probes with `PYTHONDONTWRITEBYTECODE=1` after you delete
  every `__pycache__` directory outside `.venv`. Commit your edits first: restoring a
  mutated file with `git checkout` also discards any uncommitted change to that file.
- Out of scope: tracker orchestration, SAM3 prompts, VEval/SA-FARI, plots, VACE,
  ID-Euclidean. Don't add them.
