# moteval.benchmarks

One module per supported benchmark, plus the machinery they share. For the user-facing
dataset table (sources, download tooling, on-disk layouts) see
[docs/DATASETS.md](../../docs/DATASETS.md).

- Each benchmark module (`dancetrack.py`, `sportsmot.py`, `mots20.py`, `bft.py`,
  `animaltrack.py`, `gmot40.py`, `chimpact.py`, `panaf500.py`, `uavdt.py`) defines a
  `load_<name>(root=None, split=...)` loader that reads the on-disk ground truth into a
  `MOTDataset`. `__init__.py` indexes them all in the explicit `BENCHMARKS` dict, which
  backs the public `load_dataset(name, root, split)` and the CLI's `--dataset` names.
- `motchallenge.py` holds the shared MOTChallenge-layout reader (`load_layout` +
  `MOTChallengeConfig`) that several loaders build on, plus the generic
  `load_motchallenge(root, split)` for custom standard-layout box data; `mots20.py`
  likewise exposes the generic `load_mots(root, split)` for custom mask data. Custom
  data never registers anything — it loads by path, or constructs a `MOTDataset`
  directly.
- Benchmark downloads are dev tooling in `scripts/download_benchmarks.py` (declarative
  `SPECS` + `list/status/download` subcommands), targeting `data/benchmarks/<name>` by
  default (`MOTEVAL_DATA_ROOT` or `--root` overrides).

## Adding a benchmark

1. Write `moteval/benchmarks/<name>.py` with a `load_<name>(root=None, split=...)`
   loader that returns a `MOTDataset` and declares its `Protocol` (frame convention,
   eval classes, preprocessing) — never subclass-hook preprocessing.
2. Add it to the `BENCHMARKS` dict in `moteval/benchmarks/__init__.py`.
3. Add a `DownloadSpec` to `scripts/download_benchmarks.py` (or an `unfetchable_reason`
   if the data needs manual acquisition).

## Conventions worth knowing

- GMOT-40 and ChimpACT are natively 0-indexed; their loaders keep raw frame numbers and
  declare `FrameConvention(first_frame=0)` instead of shifting.
- BFT, AnimalTrack, and GMOT-40 have no `seqinfo.ini`, so `num_timesteps` derives from
  the last annotated frame; predictions past it raise the frame-out-of-range error.
- UAVDT ignore regions (`<seq>_gt_ignore.txt`) are honored by the protocol.
