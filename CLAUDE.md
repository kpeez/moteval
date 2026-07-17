# moteval — agent guide

Issue tracker: github

MOT evaluation library: a from-scratch TrackEval rewrite that must produce **bit-identical
numbers** to official TrackEval commit `12c8791b`. Evaluation only — it never runs models.

## Source of truth

- **Spec**: `docs/agents/specs/0001-moteval-rewrite.md` (gitignored vault symlink) — approved;
  Goal/Scope/Success criteria are authoritative.
- **ADRs**: `docs/agents/adrs/0001–0003` — parity policy, canonical model + declarative
  protocol, scope boundaries. Read before proposing architecture changes.
- **Status**: GitHub issues — parent [#1](https://github.com/kpeez/moteval/issues/1),
  children #2–#20 with blocked-by ordering. Comment progress on the active issue; never
  track status in local files.

## Non-negotiables (parity policy, ADR-0001)

- Oracle = patched TrackEval `12c8791b` vendored under `tests/oracle/_trackeval/` (dev-only;
  numpy≥2 alias fixes only; every patch logged in `VENDORED.md`).
- Replicate upstream's *intentional* quirks exactly: eps-guarded thresholds
  (`similarity >= alpha - eps` in HOTA, `< threshold - eps` in CLEAR), CLEAR's `1000×`
  same-ID cost bonus, Identity's `(G+T)×(G+T)` block cost matrix with `1e10` off-diagonals,
  TrackMAP's right-to-left monotonic precision + `np.searchsorted`.
- Sole permitted numeric divergence: TrackMAP `combine_classes_det_averaged` (upstream bug;
  we implement the correct detection-weighted average).
- Fix non-numeric hazards freely: ID densification uses dicts, never `np.max(ids)+1` arrays.
- Masks are pycocotools RLE; encode from Fortran-contiguous `(h, w, n)` arrays only.

## Data model rules (ADR-0002)

- Everything converges to `MOTDataset` → frozen frame-major `SequenceData`; metrics consume
  `SequenceData` alone (precomputed per-frame similarity; J&F may also touch geometry).
- Frame-indexing is a **declared loader parameter**; out-of-range frames raise — a silent
  drop (the old track-zoo bug) is never acceptable. Regression tests must number predictions
  independently of GT.
- Per-benchmark preprocessing is a declarative `Protocol` executed by the shared engine —
  never subclass-hook preprocessing logic.

## Commands

```sh
just install   # uv sync --locked + prek hooks
just check     # ruff format, ruff check --fix, ty check
just test      # pytest (testpaths: tests/)
```

All three must pass before any PR. Scratch experiments go in gitignored `tests/temp/`.

## Layout

`moteval/data/` (model, convert, similarity, protocol) · `moteval/formats/` (MOT-txt,
MOTS-txt) · `moteval/metrics/` (base ABC + hota/clear/identity/count/jf/track_map) ·
`moteval/benchmarks/` (registry, base, per-benchmark modules) · `moteval/results.py` ·
`moteval/cli.py` · `tests/{oracle,parity,golden}/`.

## Gotchas

- `data/` is a symlink to `/data/downloads`; parity tests needing real data skip loudly when
  absent (DanceTrack val is required for the final parity gate, issue #20).
- Golden fixture oracles come from track-zoo git history: `git show f48f449:<path>` in
  <https://github.com/kpeez/track-zoo>.
- GMOT-40 and ChimpACT are natively 0-indexed (loaders shift); ChimpACT GT is COCO keyframes
  every 10th frame with linear interpolation between them, dropping `bbox_id == 23`.
- UAVDT ignore regions (`<seq>_gt_ignore.txt`) must be honored — the legacy loader ignored
  them; that was a protocol gap, not a decision.
- Out of scope (ADR-0003): tracker orchestration, SAM3 prompts, VEval/SA-FARI, plots, VACE,
  ID-Euclidean. Don't add them.
