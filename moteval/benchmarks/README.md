# moteval.benchmarks

The built-in benchmark loaders. For the user-facing dataset table (sources, download
tooling, on-disk layouts) see [docs/DATASETS.md](../../docs/DATASETS.md).

- The loaders run in Rust: one module per benchmark in `src/benchmarks/`
  (`dancetrack.rs`, `sportsmot.rs`, `mots20.rs`, `bft.rs`, `animaltrack.rs`, `gmot40.rs`,
  `chimpact.rs`, `panaf500.rs`, `uavdt.rs`), each with a `load_<name>(root, split)` that
  reads the on-disk ground truth into a `MOTDataset` and declares its `Protocol`. `root`
  is required; loaders hold no default path. `mod.rs` indexes them in the explicit
  `BENCHMARKS` table (name, loader, default split), which backs `load_dataset(name,
  root, split)` and the CLI's `--dataset` names.
- `__init__.py` is the Python side: `BENCHMARKS` (name -> loader with the benchmark's
  default split), `load_dataset`, `load_motchallenge`, `load_mots`, and
  `default_data_root()`, each a call into the extension that turns its plain data into
  the frozen dataclasses.
- `default_data_root()`: `MOTEVAL_DATA_ROOT` if set (an empty value raises), else
  `data/benchmarks`. `load_dataset` reads `default_data_root() / name` when the caller
  passes no `root`.
- `motchallenge.rs` holds the shared MOTChallenge-layout reader (`load_layout` +
  `MOTChallengeConfig`) that several loaders build on, plus the generic
  `load_motchallenge(root, split)` for custom standard-layout box data; `mots20.rs`
  likewise holds the generic `load_mots(root, split, class_id)` for custom mask data.
  Custom data never registers anything — it loads by path, or constructs a `MOTDataset`
  directly.
- Benchmark downloads are dev tooling in `scripts/download_benchmarks.py` (declarative
  `SPECS` + `list/status/download` subcommands), targeting `default_data_root() / <name>`
  unless `--root` overrides it, so downloads land where `load_dataset` reads.

## Adding a benchmark

1. Write `src/benchmarks/<name>.rs` with a `load_<name>(root, split)`
   loader that returns a `MOTDataset` and declares its `Protocol` (frame convention,
   eval classes, preprocessing) — never subclass-hook preprocessing. Errors are a
   `BenchmarkError` whose text names the file and the problem.
2. Add it to `BENCHMARKS` in `src/benchmarks/mod.rs`; the Python
   `BENCHMARKS` and the CLI pick it up from there.
3. Add a `DownloadSpec` to `scripts/download_benchmarks.py` (or an `unfetchable_reason`
   if the data needs manual acquisition).
4. Add loader tests to `tests/test_loaders.py` (through `BENCHMARKS["<name>"]`) and a
   case to `tests/test_loaders_real.py`.

## Conventions worth knowing

- GMOT-40 and ChimpACT are natively 0-indexed; their loaders keep raw frame numbers and
  declare `FrameConvention(first_frame=0)` instead of shifting.
- BFT, AnimalTrack, and GMOT-40 have no `seqinfo.ini`, so `num_timesteps` derives from
  the last annotated frame; predictions past it raise the frame-out-of-range error.
- UAVDT ignore regions (`<seq>_gt_ignore.txt`) are honored by the protocol.
