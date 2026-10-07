# moteval — agent guide

Issue tracker: Linear

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
  `SequenceData` alone (precomputed per-frame similarity; J&F may also touch geometry).
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
  protocols. Single-class JSON, CSV and table output must not change.
- `scripts/download_benchmarks.py` — dev-only benchmark downloader (`list/status/download`)
- `tests/` — flat suite (test_metrics.py, test_parity.py, test_parity_real.py, test_data.py,
  test_loaders.py, test_masks.py, test_cli.py, test_download.py), `tests/fixtures/*.json`
  (frozen TrackEval oracle numbers), `tests/scenarios.py` (shared scenario definitions),
  `tests/perturb.py` (seeded perturbed predictions), `tests/rle.py` (RLE encoding for mask
  fixtures), `tests/conftest.py` (in-memory toy dataset and a two-class dataset; the
  `toy_benchmark` and `two_class_benchmark` fixtures register them by name for a single
  test). `tests/temp/` is gitignored scratch
  for data, not tests (see Gotchas).

## Commands

```sh
just install   # uv sync --locked + prek hooks
just check     # ruff format, ruff check --fix, pyrefly check
just test      # pytest (testpaths: tests/; real-data gate deselected by default)
just test-real # slow real-data parity gate (needs the data root + fixtures)
```

All three must pass before any PR.

## Gotchas

- Parity fixtures (`tests/fixtures/*.json`) are never hand-edited — regenerate with
  `scripts/regen_parity_fixtures.py`, which clones TrackEval @ `12c8791b`, applies
  numpy>=2 alias patches, and rewrites the JSONs. The oracle datasets score one class,
  so the class-combination entries score one single-class view per class
  (`tests.scenarios.class_views`). The box entry treats a scenario's sequences as
  classes. The mask entry splits one sequence by track id (`MOTS_MULTI_CLASS_TRACKS`).
- `data/benchmarks` is a symlink to external storage holding one dir per dataset.
  `moteval.benchmarks.default_data_root()` owns the data-root rule: `MOTEVAL_DATA_ROOT`
  if set (an empty value raises), else `data/benchmarks` relative to the working
  directory. `load_dataset(name)` reads `<data root>/<name>`, `uv run
  scripts/download_benchmarks.py download <name>` writes there (`--root` overrides),
  and `just test-real` reads from it. Individual loaders take a required `root` and hold
  no default path. Parity tests needing real data skip loudly when it's absent.
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
  every `__pycache__` directory outside `.venv`.
- Out of scope: tracker orchestration, SAM3 prompts, VEval/SA-FARI, plots, VACE,
  ID-Euclidean. Don't add them.
