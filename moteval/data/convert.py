"""Build frame-major `SequenceData` from raw GT + prediction `Track` rows.

Densifies ids via a dict mapping over sorted unique raw ids (never ``max(ids)+1``
dense arrays), validates every frame against the declared `FrameConvention`, and
precomputes per-frame box-IoU similarity.
"""

import numpy as np

from moteval.data.model import FrameConvention, GtSequence, SequenceData
from moteval.data.similarity import box_iou
from moteval.formats.mot_txt import Track


def _densify(tracks: tuple[Track, ...]) -> dict[int, int]:
    unique_ids = sorted({t.track_id for t in tracks})
    return {raw: dense for dense, raw in enumerate(unique_ids)}


def _per_frame(
    tracks: tuple[Track, ...],
    id_map: dict[int, int],
    num_timesteps: int,
    convention: FrameConvention,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ids: list[list[int]] = [[] for _ in range(num_timesteps)]
    boxes: list[list[list[float]]] = [[] for _ in range(num_timesteps)]
    confs: list[list[float]] = [[] for _ in range(num_timesteps)]
    for t in tracks:
        index = convention.to_index(t.frame, num_timesteps)
        ids[index].append(id_map[t.track_id])
        boxes[index].append([t.x, t.y, t.w, t.h])
        confs[index].append(t.conf)
    id_arrays = [np.array(f, dtype=np.int64) for f in ids]
    box_arrays = [np.array(f, dtype=np.float64).reshape(-1, 4) for f in boxes]
    conf_arrays = [np.array(f, dtype=np.float64) for f in confs]
    return id_arrays, box_arrays, conf_arrays


def build_sequence_data(
    gt: GtSequence,
    pred_tracks: tuple[Track, ...],
    convention: FrameConvention,
) -> SequenceData:
    gt_map = _densify(gt.tracks)
    pred_map = _densify(pred_tracks)

    gt_ids, gt_boxes, _ = _per_frame(gt.tracks, gt_map, gt.num_timesteps, convention)
    pred_ids, pred_boxes, pred_confs = _per_frame(
        pred_tracks, pred_map, gt.num_timesteps, convention
    )
    similarity = tuple(box_iou(g, p) for g, p in zip(gt_boxes, pred_boxes, strict=True))

    return SequenceData(
        name=gt.name,
        num_timesteps=gt.num_timesteps,
        num_gt_ids=len(gt_map),
        num_pred_ids=len(pred_map),
        num_gt_dets=sum(a.shape[0] for a in gt_boxes),
        num_pred_dets=sum(a.shape[0] for a in pred_boxes),
        gt_ids=tuple(gt_ids),
        pred_ids=tuple(pred_ids),
        pred_confidences=tuple(pred_confs),
        gt_boxes=tuple(gt_boxes),
        pred_boxes=tuple(pred_boxes),
        similarity=similarity,
    )
