# moteval

Multi-object tracking evaluation — a from-scratch, typed, extensible rewrite of
[TrackEval](https://github.com/JonathonLuiten/TrackEval), producing **bit-identical numbers**
to the official reference (commit `12c8791b`) while replacing its architecture.

> **Status: under construction.** The design is settled and sliced into issues — see the
> [spec parent issue](https://github.com/kpeez/moteval/issues/1). Interfaces below are the
> approved design, not all implemented yet.

## Why

TrackEval is the de facto reference for HOTA/CLEAR/Identity, but upstream has been dormant
since 2022 (unmerged numpy≥2 fixes, no releases, Python 3.7 assumptions). moteval keeps its
numbers — published results stay comparable — and modernizes everything else.

## What it does

- **Metrics**: HOTA family, CLEAR (MOTA/MOTP/…), Identity (IDF1/…), Count, J&F, TrackMAP —
  over 2D boxes **and** RLE segmentation masks.
- **Canonical data model**: every dataset converges to one typed `MOTDataset` →
  `SequenceData` representation; metrics compute on it alone. Frame-indexing conventions are
  declared per loader and validated loudly — never silently dropped.
- **Declarative benchmark protocols**: distractor classes, ignore regions, conf-zero
  semantics are per-benchmark *declarations* executed by one shared preprocessing engine.
- **Built-in benchmarks**: DanceTrack, SportsMOT, BFT, AnimalTrack, GMOT-40, ChimpACT,
  UAVDT (ignore regions honored), PanAf500, plus a generic MOTChallenge-format adapter and
  MOTS-txt mask support. Custom datasets register a loader + protocol — no core edits.
- **API-first**: `evaluate(dataset, predictions, metrics)` returns typed per-sequence and
  combined results. CLI wraps it: `moteval run --dataset dancetrack --split val --pred <dir>`
  (console table + CSV/JSON), `moteval data download|list|status` for managed dataset roots.

## Parity guarantee

Every reported field of every shipped metric matches patched TrackEval `12c8791b` exactly —
proven in CI by golden hand-derived fixtures and side-by-side runs against a vendored oracle
on real benchmark data, cross-checked against py-motmetrics. Sole documented divergence:
TrackMAP's det-averaged combiner (upstream copy-paste bug — we ship the correct
detection-weighted average).

### Known divergences from TrackEval

- **TrackMAP `combine_classes_det_averaged`**: upstream's implementation is byte-for-byte
  identical to its class-averaged combiner — despite the name, it never actually weights by
  detections (an upstream copy-paste bug). moteval implements the intended behavior instead:
  each class's contribution is weighted by its count of non-ignored detections. The
  class-averaged combiner still matches the oracle exactly; only det-averaged diverges. See
  `moteval/metrics/track_map.py`'s `combine_classes_det_averaged` for the exact formula.

## Custom datasets

Register a zero-argument loader that converts any source format into the public data model.
The protocol declares preprocessing such as evaluated and distractor classes; predictions
remain MOTChallenge `<sequence>.txt` files.

```python
import json
from pathlib import Path

import moteval

annotations = Path("ground_truth.json")
convention = moteval.FrameConvention(name="1-indexed", first_frame=1)
protocol = moteval.Protocol(
    name="my-json",
    frame_convention=convention,
    eval_classes=(1,),
    distractor_classes=(2,),
)


@moteval.register_dataset("my-json")
def load_my_json() -> moteval.MOTDataset:
    rows = json.loads(annotations.read_text())
    tracks = tuple(
        moteval.Track(
            frame=row["frame"],
            track_id=row["id"],
            x=row["box"][0],
            y=row["box"][1],
            w=row["box"][2],
            h=row["box"][3],
            conf=1.0,
            class_id=row["class_id"],
        )
        for row in rows
    )
    sequence = moteval.GtSequence(name="video-1", num_timesteps=100, tracks=tracks)
    return moteval.MOTDataset("my-json", "validation", (sequence,), protocol)


dataset = moteval.load_dataset("my-json")
result = moteval.evaluate(
    dataset,
    "predictions/",
    [moteval.HOTA(), moteval.CLEAR(), moteval.Identity(), moteval.Count()],
)
```

## Install & develop

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
just install   # uv sync + prek hooks
just check     # ruff format + lint + ty type-check
just test      # pytest
```

Benchmark data lives under the `data/` symlink; parity tests that need real data skip loudly when it is absent.
