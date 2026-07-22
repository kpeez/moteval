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
  we implement the correct detection-weighted average).
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
- `moteval/results.py`, `moteval/cli.py`
- `scripts/download_benchmarks.py` — dev-only benchmark downloader (`list/status/download`)
- `tests/` — flat suite (test_metrics.py, test_parity.py, test_parity_real.py, test_data.py,
  test_loaders.py, test_masks.py, test_cli.py, test_download.py), `tests/fixtures/*.json`
  (frozen TrackEval oracle numbers), `tests/scenarios.py` (shared scenario definitions),
  `tests/conftest.py` (in-memory toy dataset). `tests/temp/` is gitignored scratch.

## Commands

```sh
just install   # uv sync --locked + prek hooks
just check     # ruff format, ruff check --fix, ty check
just test      # pytest (testpaths: tests/; real-data gate deselected by default)
just test-real # slow real-data parity gate (needs data/benchmarks + fixtures)
```

All three must pass before any PR.

## Gotchas

- Parity fixtures (`tests/fixtures/*.json`) are never hand-edited — regenerate with
  `scripts/regen_parity_fixtures.py`, which clones TrackEval @ `12c8791b`, applies
  numpy>=2 alias patches, and rewrites the JSONs.
- `data/benchmarks` is a symlink to external storage holding one dir per dataset;
  `uv run scripts/download_benchmarks.py download <name>` targets
  `data/benchmarks/<dataset>` by default. Parity tests needing real data skip loudly
  when it's absent.
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
- Out of scope: tracker orchestration, SAM3 prompts, VEval/SA-FARI, plots, VACE,
  ID-Euclidean. Don't add them.
