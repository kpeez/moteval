# moteval

Multi-object tracking (MOT) evaluation for Python: HOTA, CLEAR, Identity, TrackMAP, and
J&F over 2D boxes and RLE segmentation masks, with a typed data model, built-in benchmark
loaders, and a CLI.

## Installation

moteval requires Python ≥ 3.10 and is not yet on PyPI; install from GitHub:

```sh
pip install git+https://github.com/kpeez/moteval.git
```

### Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just):

```sh
git clone https://github.com/kpeez/moteval.git
cd moteval
just install   # uv sync + pre-commit hooks
just check     # ruff format + lint + pyrefly type-check
just test      # pytest
```

## Quick start

Tracker predictions are MOTChallenge `<sequence>.txt` files in one directory. Every
sequence needs a file; an empty file means no predictions for that sequence. If a file is
missing, moteval stops before it scores anything and names every missing sequence.
Evaluate a built-in benchmark by name, or point `--gt` at any directory in the standard
MOTChallenge layout (`<root>/<split>/<seq>/gt/gt.txt` + `seqinfo.ini`):

```sh
moteval run --dataset dancetrack --split val --pred path/to/tracker/output
moteval run --gt path/to/your/gt-root --split train --pred path/to/tracker/output
```

```text
seq            HOTA    DetA    AssA  MOTA    MOTP  IDSW    IDF1  Dets  GT_Dets
sequence-01  97.776  96.491     100   100  92.799     0     100    10       10
sequence-02  79.945  76.842  83.684    80  92.218     0  88.889     8       10
COMBINED     89.334  86.667  92.895    90  92.541     0  94.737    18       20
```

`--metrics hota,clear,identity,count,track_map,jf` selects metrics, `--out-csv` /
`--out-json` export every declared metric field (not just the headline columns), and
`--format mots` reads mask (MOTS-txt) ground truth. TrackMAP reports combined scores only:
its AP pools detections across all sequences, so it has no per-sequence entry.

The same from Python:

```python
import moteval

dataset = moteval.load_dataset("dancetrack", split="val")
# or, for your own data in the standard layout:
# dataset = moteval.load_motchallenge("path/to/your/gt-root", split="train")

result = moteval.evaluate(dataset, "path/to/tracker/output", [moteval.HOTA(), moteval.CLEAR()])
print(result.combined["CLEAR"]["MOTA"])
```

Ground truth in any other format converts by constructing a `moteval.MOTDataset` — the
tracks plus a `Protocol` declaring the frame convention and evaluated classes — and
calling `evaluate` on it.

### Multi-class evaluation

Every built-in benchmark evaluates one class. A `Protocol` whose `eval_classes` lists
several class ids makes `evaluate` score each class on its own and then combine the
classes in the two ways TrackEval does:

- Box predictions give their class in column 8 of each MOTChallenge row (as `gt.txt`
  does). moteval reads that column only for multi-class protocols, so single-class
  prediction files that hold `-1` there keep working. In a multi-class run, a row with
  no column 8 or a negative class raises. MOTS rows always carry their class.
- `EvaluationResult.per_class` maps each class id to that class's `per_sequence` and
  `combined` scores. `class_averaged` is the plain mean over classes;
  `det_averaged` pools the detections of all classes. In a multi-class run,
  `per_sequence` and `combined` are empty. In a single-class run, the three new fields
  are empty, and the result, JSON, CSV and table are the same as before.
- JSON adds `per_class` (class id as a string), `class_averaged` and `det_averaged`.
  CSV columns become `class,seq,metric,field,value`: one block per class id (each
  sequence, then `COMBINED`), then `class_averaged` and `det_averaged` rows with
  `seq = COMBINED`. The table gains the same leading `class` column and rows.

## Metrics

| CLI name | Metric | Fields |
| --- | --- | --- |
| `hota` | HOTA | HOTA, DetA, AssA, DetRe, DetPr, AssRe, AssPr, LocA, OWTA (arrays over 19 IoU thresholds) + HOTA(0), LocA(0), HOTALocA(0) |
| `clear` | CLEAR | MOTA, MOTP, MODA, MT/PT/ML, IDSW, Frag, CLR_Re/Pr/F1, sMOTA, MOTAL, … |
| `identity` | Identity | IDF1, IDR, IDP, IDTP, IDFN, IDFP |
| `count` | Count | Dets, GT_Dets, IDs, GT_IDs |
| `track_map` | TrackMAP | AP/AR over IoU thresholds, split by track area and length |
| `jf` | J&F (masks) | J-Mean, J-Recall, J-Decay, F-Mean, F-Recall, F-Decay, J&F |

## Supported benchmarks

| Benchmark | Domain | Ground truth | Default split | Auto-download |
| --- | --- | --- | --- | --- |
| [DanceTrack](https://github.com/DanceTrack/DanceTrack) | persons | MOT boxes | `val` | ✓ |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | persons | MOT boxes | `val` | ✓ |
| [MOTS20](https://motchallenge.net/data/MOTS/) | persons | MOTS masks | `train` | ✓ |
| [BFT](https://github.com/George-Zhuang/NetTrack) | birds | MOT boxes | `val` | ✓ |
| [AnimalTrack](https://hengfan2010.github.io/projects/AnimalTrack/evaluation.html) | animals | MOT boxes | `all` | ✓ |
| [GMOT-40](https://github.com/Spritea/GMOT40) | generic | MOT boxes | `test` | ✓ |
| [PanAf500](https://obrookes.github.io/panaf.github.io/) | great apes | COCO-style JSON | `validation` | ✓ |
| [ChimpACT](https://github.com/ShirleyMaxx/ChimpACT) | chimpanzees | COCO-style JSON | `val` | manual |
| [UAVDT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5) | vehicles | MOT boxes + ignore regions | `all` | ✓ |

`uv run scripts/download_benchmarks.py download <name>` fetches annotations into
`data/benchmarks/<name>` (frames/videos are never needed for scoring), and
`moteval.load_dataset(name)` / `moteval run --dataset <name>` read from the same place.
Set `MOTEVAL_DATA_ROOT` to move that root for both; `--root` (download script),
`--gt` (CLI) or `root=` (Python) overrides it for one call. Sources and
on-disk layouts: [docs/DATASETS.md](docs/DATASETS.md).

## Parity with TrackEval

moteval is a from-scratch rewrite of [TrackEval](https://github.com/JonathonLuiten/TrackEval)
(commit `12c8791b`) that reproduces its numbers exactly, so published results stay
comparable. This is enforced by tests rather than claimed: the suite asserts exact
equality against frozen TrackEval outputs on synthetic scenarios
(`tests/fixtures/*.json`, regenerated only by `scripts/regen_parity_fixtures.py`, which
runs the pinned upstream), and `just test-real` repeats the bit-identical check on real
DanceTrack, SportsMOT, and MOTS20 data.

One documented divergence: upstream's TrackMAP `combine_classes_det_averaged` is a
copy-paste of its class-averaged combiner and never actually weights by detections (an
upstream bug); moteval implements the intended detection-weighted average. It affects
only TrackMAP's `det_averaged` result in a multi-class run. Everything else matches to
the last bit, including upstream's intentional quirks and every other metric's
class-averaged and det-averaged results.
