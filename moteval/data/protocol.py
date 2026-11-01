"""Declarative per-benchmark preprocessing `Protocol`.

A `Protocol` is a value object: it declares *what* preprocessing a benchmark
needs (evaluated classes, distractor classes, ignore-region IoA threshold,
GT conf-zero semantics, frame convention) but never *how* to run it. One shared
engine in the Rust extension (``src/protocol.rs``) executes any Protocol
on each frame, replicating TrackEval ``get_preprocessed_seq_data`` semantics (commit
12c8791b) exactly:

    1. keep only predictions of the class under evaluation;
    2. Hungarian-match those predictions against *all* GT (distractor classes
       included) and drop predictions matched to a distractor-class GT box;
    3. drop unmatched predictions whose IoA with an ignore region exceeds the
       threshold;
    4. keep only GT of the evaluated class, dropping conf-zero rows.

The engine is geometry-agnostic: it consumes precomputed ``similarity`` (GT x pred)
and ``ignore_ioa`` (pred x ignore) matrices, so masks slot in behind the same call
by swapping how those matrices are built during conversion.
"""

from dataclasses import dataclass

from moteval.data.model import FrameConvention


@dataclass(frozen=True)
class Protocol:
    """Declarative preprocessing rules for one benchmark.

    ``eval_classes`` are the class ids scored (one `SequenceData` view per class).
    ``distractor_classes`` are matched against but never evaluated. Thresholds carry
    TrackEval's canonical ``0.5`` defaults; ``drop_zero_conf_gt`` toggles the
    zero-marked GT exclusion (off only for preproc-free benchmarks like MOT15).
    ``matching_fill`` is the score below-threshold pairs get during the Hungarian
    matching step: upstream's box path zeroes them while its MOTS path sets
    ``-10000``, and the two can tie-break differently, so the value is a declared
    protocol parameter replicated exactly. On valid mask input, for a protocol with
    no distractor classes (MOTS20 declares none), the MOTS fill cannot change a
    result. GT and ignore masks never overlap (both sides reject such input), so a
    prediction at or above the 0.5 matching threshold to a GT mask has at least
    half its area in that mask. It never passes the ignore test, whether matched or
    not. A prediction with no such GT mask is unmatched under either fill.
    """

    name: str
    frame_convention: FrameConvention
    eval_classes: tuple[int, ...]
    distractor_classes: tuple[int, ...] = ()
    distractor_iou_threshold: float = 0.5
    ignore_iou_threshold: float = 0.5
    drop_zero_conf_gt: bool = True
    matching_fill: float = 0.0
