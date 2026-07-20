# Benchmark datasets

Ground-truth annotations for every benchmark moteval can score. `moteval data download
<name>` fetches a dataset into `data/benchmarks/<name>` (override with `--root` or
`MOTEVAL_DATA_ROOT`); `moteval data list` / `moteval data status` show availability and
on-disk state. moteval downloads **annotations only** — frames/videos are never needed
for scoring.

| Dataset | Domain | Annotations | Source |
| --- | --- | --- | --- |
| DanceTrack | persons | MOT boxes | [repo](https://github.com/DanceTrack/DanceTrack) · [HuggingFace](https://huggingface.co/datasets/noahcao/dancetrack) |
| SportsMOT | persons | MOT boxes | [repo](https://github.com/MCG-NJU/SportsMOT) · [HuggingFace](https://huggingface.co/datasets/MCG-NJU/SportsMOT) |
| MOTS20 | persons | MOTS masks | [MOTChallenge](https://motchallenge.net/data/MOTS/) |
| BFT (Bird Flock Tracking) | animals | MOT boxes | [NetTrack repo](https://github.com/George-Zhuang/NetTrack) · [gdrive](https://drive.google.com/drive/folders/140mPnOVZY-2apH76at9yYuVGIDWOvsH_) |
| AnimalTrack | animals | MOT boxes | [project page](https://hengfan2010.github.io/projects/AnimalTrack/evaluation.html) · [gdrive](https://drive.google.com/drive/folders/1P0oaPRruthyALztjJW8nbOegOpU_szew) |
| GMOT-40 | animals | MOT boxes | [repo](https://github.com/Spritea/GMOT40) · [releases](https://github.com/Spritea/GMOT40/releases/tag/v0.1) |
| PanAf500 | animals | COCO-style JSON | [project page](https://obrookes.github.io/panaf.github.io/) · [data.bris](https://data.bris.ac.uk/data/dataset/1h73erszj3ckn2qjwm4sqmr2wt) |
| ChimpACT | animals | COCO-style JSON | [repo](https://github.com/ShirleyMaxx/ChimpACT) — **manual**: gated Google Form, no programmatic artifact |
| UAVDT | vehicles | MOT boxes + ignore regions | [website](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5) · gdrive (see `download.py`) |

## Managed layout

`moteval data download` verifies each dataset against the exact layout its loader reads
(the `expected_layout` in `moteval/benchmarks/download.py`):

```
data/benchmarks/
  dancetrack/  val/<seq>/{gt/gt.txt,seqinfo.ini}            (train/test also extracted)
  sportsmot/   val/<seq>/{gt/gt.txt,seqinfo.ini}
  mots20/      train/<seq>/{gt/gt.txt,seqinfo.ini}
  bft/         annotations_mot/<split>/<seq>.txt
  animaltrack/ gt_all/<seq>_gt.txt
  gmot40/      track_label/<seq>.txt
  panaf500/    annotations/<split>/<video>.json
  chimpact/    ChimpACT_release_v1/labels/<clip>.json       (manual placement)
  uavdt/       UAV-benchmark-MOTD_v1.0/GT/<seq>_gt.txt      (+ <seq>_gt_ignore.txt)
```

## Format notes

- MOT box rows are MOTChallenge txt: `frame, id, x, y, w, h, conf, class, visibility`
  (tlwh, absolute pixels). MOTS rows are `frame, id, class_id, h, w, rle`.
- GMOT-40 and ChimpACT are natively **0-indexed**; their loaders keep raw frame numbers
  and declare `FrameConvention(first_frame=0)`.
- BFT, AnimalTrack, and GMOT-40 ship no `seqinfo.ini`, so sequence length derives from
  the last annotated frame; predictions past it raise a frame-out-of-range error.
- UAVDT's `<seq>_gt_ignore.txt` regions are honored during preprocessing.
- ChimpACT requires requesting access via the official repository's form, then placing
  the release's COCO JSON files at the path above; `moteval data status` will report it
  `present` once the layout matches.
