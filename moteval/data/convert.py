"""Build frame-major `SequenceData` from raw GT + prediction `Track` rows.

Raw per-frame arrays are assembled first, the declarative preprocessing engine
(`preprocess_frame`) filters them for the class under evaluation, and only the
survivors are densified via a dict mapping over sorted unique raw ids (never
``max(ids)+1`` dense arrays). Every frame is validated against the declared
`FrameConvention`; per-frame box-IoU similarity and ignore-region IoA are
precomputed so the engine stays geometry-agnostic.
"""

import numpy as np

from moteval.data.model import FrameConvention, GtSequence, SequenceData
from moteval.data.protocol import PreprocessedFrame, Protocol, RawFrame, preprocess_frame
from moteval.data.similarity import box_ioa, box_iou
from moteval.formats.mot_txt import Track


def _densify(ids: list[int]) -> dict[int, int]:
    return {raw: dense for dense, raw in enumerate(sorted(set(ids)))}


def _bin_frames(
    tracks: tuple[Track, ...], num_timesteps: int, convention: FrameConvention
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ids: list[list[int]] = [[] for _ in range(num_timesteps)]
    classes: list[list[int]] = [[] for _ in range(num_timesteps)]
    confs: list[list[float]] = [[] for _ in range(num_timesteps)]
    boxes: list[list[list[float]]] = [[] for _ in range(num_timesteps)]
    for t in tracks:
        index = convention.to_index(t.frame, num_timesteps)
        ids[index].append(t.track_id)
        classes[index].append(t.class_id)
        confs[index].append(t.conf)
        boxes[index].append([t.x, t.y, t.w, t.h])
    id_arrays = [np.array(f, dtype=np.int64) for f in ids]
    class_arrays = [np.array(f, dtype=np.int64) for f in classes]
    conf_arrays = [np.array(f, dtype=np.float64) for f in confs]
    box_arrays = [np.array(f, dtype=np.float64).reshape(-1, 4) for f in boxes]
    return id_arrays, class_arrays, conf_arrays, box_arrays


def build_sequence_data(
    gt: GtSequence,
    pred_tracks: tuple[Track, ...],
    protocol: Protocol,
    cls_id: int,
) -> SequenceData:
    convention = protocol.frame_convention
    gt_ids, gt_classes, gt_conf, gt_boxes = _bin_frames(gt.tracks, gt.num_timesteps, convention)
    pred_ids, pred_classes, pred_conf, pred_boxes = _bin_frames(
        pred_tracks, gt.num_timesteps, convention
    )
    _, _, _, ignore_boxes = _bin_frames(gt.ignore_regions, gt.num_timesteps, convention)

    frames = [
        preprocess_frame(
            RawFrame(
                gt_ids=gt_ids[t],
                gt_classes=gt_classes[t],
                gt_conf=gt_conf[t],
                gt_boxes=gt_boxes[t],
                pred_ids=pred_ids[t],
                pred_classes=pred_classes[t],
                pred_confidences=pred_conf[t],
                pred_boxes=pred_boxes[t],
                similarity=box_iou(gt_boxes[t], pred_boxes[t]),
                ignore_ioa=box_ioa(pred_boxes[t], ignore_boxes[t]),
            ),
            protocol,
            cls_id,
        )
        for t in range(gt.num_timesteps)
    ]

    return _freeze(gt.name, gt.num_timesteps, frames)


def _freeze(name: str, num_timesteps: int, frames: list[PreprocessedFrame]) -> SequenceData:
    gt_map = _densify([i for f in frames for i in f.gt_ids.tolist()])
    pred_map = _densify([i for f in frames for i in f.pred_ids.tolist()])
    gt_ids = tuple(np.array([gt_map[i] for i in f.gt_ids.tolist()], dtype=np.int64) for f in frames)
    pred_ids = tuple(
        np.array([pred_map[i] for i in f.pred_ids.tolist()], dtype=np.int64) for f in frames
    )
    return SequenceData(
        name=name,
        num_timesteps=num_timesteps,
        num_gt_ids=len(gt_map),
        num_pred_ids=len(pred_map),
        num_gt_dets=sum(f.gt_ids.shape[0] for f in frames),
        num_pred_dets=sum(f.pred_ids.shape[0] for f in frames),
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        pred_confidences=tuple(f.pred_confidences for f in frames),
        gt_boxes=tuple(f.gt_boxes for f in frames),
        pred_boxes=tuple(f.pred_boxes for f in frames),
        similarity=tuple(f.similarity for f in frames),
    )
