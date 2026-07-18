"""Build frame-major `SequenceData` from raw GT + prediction rows (boxes or masks).

Raw per-frame arrays are assembled first, the declarative preprocessing engine
(`preprocess_frame`) filters them for the class under evaluation, and only the
survivors are densified via a dict mapping over sorted unique raw ids (never
``max(ids)+1`` dense arrays). Every frame is validated against the declared
`FrameConvention`; per-frame similarity (box IoU or mask IoU) and ignore-region
IoA are precomputed so the engine stays geometry-agnostic.

The mask path replicates TrackEval's MOTS loading exactly: per-frame ignore
masks are unioned into a single RLE before the IoA test, and overlapping masks
within a frame (GT dets + ignore region together; predictions on their own) are
rejected loudly, as upstream rejects them.
"""

import numpy as np

from moteval.data.model import (
    BoxGeometry,
    FrameConvention,
    GtSequence,
    MaskGeometry,
    MaskGtSequence,
    RleMask,
    SequenceData,
)
from moteval.data.protocol import PreprocessedFrame, Protocol, RawFrame, preprocess_frame
from moteval.data.similarity import (
    box_ioa,
    box_iou,
    mask_ioa,
    mask_iou,
    masks_overlap,
    merge_masks,
)
from moteval.formats.mot_txt import Track
from moteval.formats.mots_txt import MaskTrack


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
                gt_dets=gt_boxes[t],
                pred_ids=pred_ids[t],
                pred_classes=pred_classes[t],
                pred_confidences=pred_conf[t],
                pred_dets=pred_boxes[t],
                similarity=box_iou(gt_boxes[t], pred_boxes[t]),
                ignore_ioa=box_ioa(pred_boxes[t], ignore_boxes[t]),
            ),
            protocol,
            cls_id,
        )
        for t in range(gt.num_timesteps)
    ]

    return _freeze(gt.name, gt.num_timesteps, frames, kind="box")


def _rle(track: MaskTrack) -> RleMask:
    return {"size": [track.img_h, track.img_w], "counts": track.rle.encode()}


def _bin_mask_frames(
    tracks: tuple[MaskTrack, ...], num_timesteps: int, convention: FrameConvention
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ids: list[list[int]] = [[] for _ in range(num_timesteps)]
    classes: list[list[int]] = [[] for _ in range(num_timesteps)]
    rles: list[list[RleMask]] = [[] for _ in range(num_timesteps)]
    for t in tracks:
        index = convention.to_index(t.frame, num_timesteps)
        ids[index].append(t.track_id)
        classes[index].append(t.class_id)
        rles[index].append(_rle(t))
    id_arrays = [np.array(f, dtype=np.int64) for f in ids]
    class_arrays = [np.array(f, dtype=np.int64) for f in classes]
    rle_arrays = [_object_array(f) for f in rles]
    return id_arrays, class_arrays, rle_arrays


def _object_array(rles: list[RleMask]) -> np.ndarray:
    arr = np.empty(len(rles), dtype=object)
    for i, rle in enumerate(rles):
        arr[i] = rle
    return arr


def build_mask_sequence_data(
    gt: MaskGtSequence,
    pred_tracks: tuple[MaskTrack, ...],
    protocol: Protocol,
    cls_id: int,
) -> SequenceData:
    convention = protocol.frame_convention
    gt_ids, gt_classes, gt_rles = _bin_mask_frames(gt.tracks, gt.num_timesteps, convention)
    pred_ids, pred_classes, pred_rles = _bin_mask_frames(pred_tracks, gt.num_timesteps, convention)
    _, _, ignore_rles = _bin_mask_frames(gt.ignore_regions, gt.num_timesteps, convention)

    frames = []
    for t in range(gt.num_timesteps):
        # TrackEval rejects overlapping masks within a frame: GT dets share the
        # frame with the ignore region; predictions are checked on their own.
        if masks_overlap(list(gt_rles[t]) + list(ignore_rles[t])):
            raise ValueError(f"sequence {gt.name!r}: overlapping GT masks at timestep {t}")
        if masks_overlap(list(pred_rles[t])):
            raise ValueError(f"sequence {gt.name!r}: overlapping predicted masks at timestep {t}")
        # Ignore masks merge into one union RLE, exactly as upstream MOTS does;
        # with no ignore rows the IoA matrix is (n, 0) and the engine skips it.
        if len(ignore_rles[t]):
            merged_ignore = [merge_masks(list(ignore_rles[t]))]
            ignore_ioa = mask_ioa(list(pred_rles[t]), merged_ignore)
        else:
            ignore_ioa = np.zeros((len(pred_rles[t]), 0), dtype=np.float64)
        frames.append(
            preprocess_frame(
                RawFrame(
                    gt_ids=gt_ids[t],
                    gt_classes=gt_classes[t],
                    gt_conf=np.ones(len(gt_ids[t]), dtype=np.float64),
                    gt_dets=gt_rles[t],
                    pred_ids=pred_ids[t],
                    pred_classes=pred_classes[t],
                    pred_confidences=np.ones(len(pred_ids[t]), dtype=np.float64),
                    pred_dets=pred_rles[t],
                    similarity=mask_iou(list(gt_rles[t]), list(pred_rles[t])),
                    ignore_ioa=ignore_ioa,
                ),
                protocol,
                cls_id,
            )
        )

    return _freeze(gt.name, gt.num_timesteps, frames, kind="mask")


def _freeze(
    name: str, num_timesteps: int, frames: list[PreprocessedFrame], kind: str
) -> SequenceData:
    gt_map = _densify([i for f in frames for i in f.gt_ids.tolist()])
    pred_map = _densify([i for f in frames for i in f.pred_ids.tolist()])
    gt_ids = tuple(np.array([gt_map[i] for i in f.gt_ids.tolist()], dtype=np.int64) for f in frames)
    pred_ids = tuple(
        np.array([pred_map[i] for i in f.pred_ids.tolist()], dtype=np.int64) for f in frames
    )
    geometry_cls = {"box": BoxGeometry, "mask": MaskGeometry}[kind]
    geometry = geometry_cls(
        gt=tuple(f.gt_dets for f in frames), pred=tuple(f.pred_dets for f in frames)
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
        geometry=geometry,
        similarity=tuple(f.similarity for f in frames),
    )
