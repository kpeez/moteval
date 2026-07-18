# Vendored TrackEval oracle

`_trackeval/` is a **dev-only** copy of official TrackEval, used as the parity oracle for
moteval's metric tests (ADR-0001). It is never imported by the shipped `moteval` package and
is excluded from the wheel and sdist (`[tool.hatch.build.targets.*]`), from ruff
(`extend-exclude`), and from ty (`[tool.ty.src] exclude`).

## Provenance

- **Upstream**: <https://github.com/JonathonLuiten/TrackEval>
- **Commit**: `12c8791b303e0a0b50f753af204249e622d0281a`
- **Commit date**: 2022-11-29
- **Vendored**: 2026-07-17
- **Scope**: only the `trackeval/` Python package was copied (as `_trackeval/`), plus upstream
  `LICENSE`. Repo docs, scripts, and `minimal_examples` were not vendored.

## Patches

Patches are **numpy>=2 alias fixes only**. Every change replaces a numpy alias that was removed
in numpy 1.24 / 2.0 with the exact builtin it aliased, so no computed number changes:

- `np.float` → `float`  (dtype float64, identical)
- `np.int`   → `int`    (dtype int_, identical)
- `np.bool`  → `bool`   (dtype bool, identical)

Applied across the package with word-boundary matching (never touching `np.float64`,
`np.int32`, `np.bool_`, `np.floating`, etc.). Occurrence counts per file:

| File | `np.float`→`float` | `np.int`→`int` | `np.bool`→`bool` |
| --- | --- | --- | --- |
| `baselines/stp.py` | | 2 | |
| `datasets/bdd100k.py` | | 2 | 2 |
| `datasets/burst_helpers/burst_base.py` | | 3 | 2 |
| `datasets/burst_helpers/burst_ow_base.py` | | 3 | 2 |
| `datasets/davis.py` | | 2 | 1 |
| `datasets/head_tracking_challenge.py` | 1 | 3 | |
| `datasets/kitti_2d_box.py` | 2 | 3 | 2 |
| `datasets/kitti_mots.py` | | 2 | 2 |
| `datasets/mot_challenge_2d_box.py` | 1 | 3 | |
| `datasets/mots_challenge.py` | | 2 | 2 |
| `datasets/person_path_22.py` | 2 | 3 | |
| `datasets/rob_mots.py` | | 3 | 2 |
| `datasets/tao.py` | | 3 | 2 |
| `datasets/tao_ow.py` | | 3 | 2 |
| `datasets/youtube_vis.py` | | 2 | 2 |
| `metrics/hota.py` | 5 | | |
| `metrics/identity.py` | | 3 | |
| `metrics/j_and_f.py` | | | 1 |
| `metrics/track_map.py` | 6 | | |

No other patches. In particular, no algorithmic, threshold, eps-guard, or cost-matrix code was
touched — the oracle reproduces upstream's numbers exactly.

## Notes for callers

- Use `tests/oracle/runner.py` (`run_mot_challenge`) to invoke the oracle; it inserts
  `tests/oracle/` onto `sys.path` and imports `_trackeval` as a top-level package.
- Importing `_trackeval.datasets` prints
  `Error importing BURST due to missing underlying dependency: No module named 'trackeval'`.
  This is upstream's own `try/except` fallback: `burst.py`/`burst_ow.py` use absolute
  `import trackeval` imports that do not resolve under the vendored `_trackeval` name. BURST is
  not used by moteval; the message is cosmetic and the rest of the package loads normally.
