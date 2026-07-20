# moteval.benchmarks

One module per supported benchmark, plus the machinery they share.

- Each benchmark module (`dancetrack.py`, `sportsmot.py`, `mots20.py`, `bft.py`,
  `animaltrack.py`, `gmot40.py`, `chimpact.py`, `panaf500.py`, `uavdt.py`) defines a
  loader that reads the on-disk ground truth into a `MOTDataset` and registers it by
  name via `moteval.data.registry.register_dataset`. `motchallenge.py` holds the shared
  MOTChallenge-layout reader that several loaders build on.
- `download.py` declares where each benchmark's data comes from (`SPECS`) and implements
  `moteval data list/status/download`. Downloads land under `data/benchmarks/<name>` by
  default (`MOTEVAL_DATA_ROOT` or `--root` overrides).

## Adding a benchmark

1. Write `moteval/benchmarks/<name>.py` with a `@register_dataset("<name>")` loader that
   returns a `MOTDataset` and declares its `Protocol` (frame convention, eval classes,
   preprocessing) — never subclass-hook preprocessing.
2. Import the module for side effect in `moteval/__init__.py`.
3. Add a `DownloadSpec` to `download.py` (or an `unfetchable_reason` if the data needs
   manual acquisition).

## Conventions worth knowing

- GMOT-40 and ChimpACT are natively 0-indexed; their loaders keep raw frame numbers and
  declare `FrameConvention(first_frame=0)` instead of shifting.
- BFT, AnimalTrack, and GMOT-40 have no `seqinfo.ini`, so `num_timesteps` derives from
  the last annotated frame; predictions past it raise the frame-out-of-range error.
- UAVDT ignore regions (`<seq>_gt_ignore.txt`) are honored by the protocol.
