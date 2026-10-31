# moteval — agent guide

Issue tracker: Linear. When Linear's free issue limit blocks a new issue, file it in
GitHub Issues on this repo instead and close it with `Fixes #N`.

MOT evaluation library: a from-scratch TrackEval rewrite that must produce **bit-identical
numbers** to official TrackEval commit `12c8791b`. Evaluation only — it never runs models.

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

## Layout

- `moteval/data/` — model, convert, similarity, protocol
- `moteval/formats.py` — MOT box rows (`Track`) and MOTS mask rows (`MaskTrack`)
- `moteval/metrics/` — base ABC + hota/clear/identity/count/jf/track_map
- `moteval/benchmarks/` — one loader module per benchmark; `__init__.py` holds the
  explicit `BENCHMARKS` dict + `load_dataset(name, root, split)` (no registration —
  custom data loads by path via `load_motchallenge`/`load_mots` or builds a `MOTDataset`
  directly); see that directory's `README.md` for loader conventions
- `moteval/eval.py` (`evaluate`), `moteval/results.py`, `moteval/cli.py`. `evaluate`
  scores each class in `protocol.eval_classes`. A single-class run fills
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
- `scripts/download_benchmarks.py` — dev-only benchmark downloader (`list/status/download`)
- `scripts/reference_results.py` — scores the nine benchmarks with seeded predictions; writes
  per-sequence and combined JSON and the reference fixture
- `tests/` — flat suite (test_metrics.py, test_parity.py, test_parity_real.py, test_data.py,
  test_loaders.py, test_loaders_real.py, test_reference_real.py, test_masks.py, test_cli.py, test_download.py,
  test_bindings.py: the Rust extension against the Python code it replaces),
  `tests/fixtures/*.json`
  (frozen TrackEval oracle numbers, except `reference_benchmarks.json`), `tests/scenarios.py` (shared scenario definitions),
  `tests/perturb.py` (seeded perturbed predictions), `tests/rle.py` (RLE encoding for mask
  fixtures), `tests/conftest.py` (in-memory toy dataset and a two-class dataset, with
  `write_perfect_predictions` and `write_two_class_predictions` to write their prediction
  files; the `toy_benchmark` and `two_class_benchmark` fixtures register them by name for a
  single test). `tests/temp/` is gitignored scratch
  for data, not tests (see Gotchas).
- `rust/` — the Rust port in progress: a Cargo workspace (`moteval` crate, `moteval-py`
  PyO3 crate), dump scripts in `rust/scripts/`, and test data dumped from Python in
  `rust/testdata/` (never hand-edited). `rust/README.md` has the layout, commands, and
  rules; CI's `rust` job runs `cargo fmt --check`, `cargo clippy -D warnings`, and
  `cargo test`. maturin is the build backend: `uv sync` compiles `moteval-py` into the
  extension `moteval._moteval` (stubs in `moteval/_moteval.pyi`), so every Python job
  needs Rust, and the `wheels` workflow builds abi3 wheels for Linux x86_64 and macOS
  arm64. The `moteval` console script still runs the Python CLI until the port is
  integrated.

## Commands

```sh
mise run install   # uv sync --locked, prek hooks (main checkout only)
mise run check     # ruff format, ruff check --fix, pyrefly check
mise run test      # pytest (testpaths: tests/; real-data gate deselected by default)
mise run test-real # slow real-data gate: parity + raw-GT loader checks (needs the data root)
```

`check`, `test`, and `test-real` must pass before any PR.

`mise.toml` pins uv, rust, and hyperfine and holds the tasks. uv pins Python through
`.python-version`; mise does not. Run `mise run install` first in every fresh clone or git
worktree: the venv is an editable install, and it rebuilds the Rust extension when a
Rust source changes (`[tool.uv] cache-keys`). Host-only settings,
such as `MOTEVAL_DATA_ROOT`, go in the gitignored `mise.local.toml`. `mise run` trusts its own
config; run `mise trust` only to activate that env in your shell. `mise run install`
installs the git hooks only in the main checkout, because a worktree shares `.git/hooks`.

## Gotchas

- Parity fixtures (`tests/fixtures/*.json`) are never hand-edited — regenerate with
  `scripts/regen_parity_fixtures.py`, which clones TrackEval @ `12c8791b`, applies
  numpy>=2 alias patches, and rewrites the JSONs. The oracle datasets score one class,
  so the class-combination entries score one single-class view per class
  (`tests.scenarios.class_views`). The box entry treats a scenario's sequences as
  classes. The mask entry splits one sequence by track id (`MOTS_MULTI_CLASS_TRACKS`).
- `tests/fixtures/reference_benchmarks.json` is not an oracle fixture. It freezes moteval's
  own combined results (every applicable metric, floats exact) on all nine local
  benchmarks and each split with ground truth, scored against seeded `tests/perturb.py`
  predictions. The Rust port is checked against it, so it changes only when a metric
  changes on purpose. Never hand-edit it. Regenerate with
  `MOTEVAL_DATA_ROOT=<root> uv run python scripts/reference_results.py --out-dir artifacts/reference --reference-fixture tests/fixtures/reference_benchmarks.json`;
  `tests/test_reference_real.py` (`real_data`, bit for bit, `np.array_equal` with
  `equal_nan`) checks it, and skips loudly when a dataset is absent.
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
