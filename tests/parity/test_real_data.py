"""Final real-data parity gate (#20): moteval vs the TrackEval oracle side-by-side
on DanceTrack val, SportsMOT val, and a MOTS20 mask sequence, with seeded,
independently-numbered perturbed predictions.

Each test skips loudly when its dataset is absent from the managed data root,
naming the `moteval data download` command that fetches it. The gate counts as
DONE only when all three have actually run green locally (a fully-skipped run
does not count) — see issue #20 for the recorded evidence.
"""

from pathlib import Path

import numpy as np
import pytest

from moteval import CLEAR, HOTA, Count, Identity, evaluate
from moteval.benchmarks.dancetrack import load_dancetrack
from moteval.benchmarks.mots20 import load_mots20
from moteval.benchmarks.sportsmot import load_sportsmot
from moteval.formats.mot_txt import write_mot
from moteval.formats.mots_txt import write_mots
from moteval.metrics.jf import JAndF
from tests.oracle.runner import run_mot_challenge, run_mots_challenge
from tests.parity.perturb import perturb_box_tracks, perturb_mask_tracks

DATA_ROOT = Path("data/benchmarks")
SEED = 20260718
BOX_METRICS = ("HOTA", "CLEAR", "Identity", "Count")


def _require(root: Path, benchmark: str, expected_sequences: int) -> None:
    if not root.is_dir() or not any(root.iterdir()):
        pytest.skip(
            f"SKIPPING REAL-DATA PARITY GATE: {benchmark} not found under {root} — "
            f"fetch it with `moteval data download {benchmark}`"
        )
    # A split that is still syncing (e.g. cloud-storage hydration in progress)
    # skips loudly rather than evaluating a partial dataset.
    seq_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    incomplete = [
        p.name
        for p in seq_dirs
        if not (p / "gt" / "gt.txt").is_file() or not (p / "seqinfo.ini").is_file()
    ]
    if len(seq_dirs) < expected_sequences or incomplete:
        pytest.skip(
            f"SKIPPING REAL-DATA PARITY GATE: {benchmark} under {root} looks incomplete "
            f"({len(seq_dirs)}/{expected_sequences} sequence dirs; missing gt/seqinfo: "
            f"{incomplete[:5]}) — still downloading? Re-run once "
            f"`moteval data download {benchmark}` has finished."
        )


def _assert_metrics_equal(ours: dict, oracle: dict, metrics: tuple[str, ...]) -> None:
    for metric in metrics:
        for field, oracle_value in oracle[metric].items():
            if field not in ours[metric]:
                continue
            ours_value = ours[metric][field]
            assert np.array_equal(
                np.asarray(ours_value, dtype=float),
                np.asarray(oracle_value, dtype=float),
                equal_nan=True,
            ), f"{metric}.{field}: {ours_value} != {oracle_value}"


def _run_box_gate(dataset, gt_root: Path, tmp_path: Path) -> None:
    pred_dir = tmp_path / "trackers" / "oracle" / "data"
    seq_lengths = {}
    for i, seq in enumerate(dataset.sequences):
        preds = perturb_box_tracks(
            seq.tracks, seq.num_timesteps, dataset.protocol.frame_convention, seed=SEED + i
        )
        write_mot(pred_dir / f"{seq.name}.txt", preds)
        seq_lengths[seq.name] = seq.num_timesteps

    ours = evaluate(dataset, pred_dir, [HOTA(), CLEAR(), Identity(), Count()]).combined
    oracle = run_mot_challenge(gt_root, tmp_path / "trackers", seq_lengths)
    _assert_metrics_equal(ours, oracle, BOX_METRICS)


def test_real_data_parity_dancetrack_val(tmp_path):
    root = DATA_ROOT / "dancetrack"
    _require(root / "val", "dancetrack", expected_sequences=25)
    dataset = load_dancetrack(root=root, split="val")
    assert len(dataset.sequences) > 0
    _run_box_gate(dataset, root / "val", tmp_path)


def test_real_data_parity_sportsmot_val(tmp_path):
    root = DATA_ROOT / "sportsmot"
    _require(root / "val", "sportsmot", expected_sequences=45)
    dataset = load_sportsmot(root=root, split="val")
    assert len(dataset.sequences) > 0
    _run_box_gate(dataset, root / "val", tmp_path)


def test_real_data_parity_mots20_mask_sequence(tmp_path):
    pytest.importorskip("cv2", reason="oracle JAndF needs cv2 (dev dependency)")
    pytest.importorskip("skimage", reason="oracle JAndF needs scikit-image (dev dependency)")
    root = DATA_ROOT / "mots20"
    _require(root / "train", "mots20", expected_sequences=4)
    dataset = load_mots20(root=root, split="train")
    assert len(dataset.sequences) > 0

    # One mask sequence suffices per the spec; take the first, keep the run bounded.
    seq = dataset.sequences[0]
    dataset = type(dataset)(
        name=dataset.name, split=dataset.split, sequences=(seq,), protocol=dataset.protocol
    )
    pred_dir = tmp_path / "trackers" / "oracle" / "data"
    preds = perturb_mask_tracks(
        seq.tracks, seq.num_timesteps, dataset.protocol.frame_convention, seed=SEED
    )
    write_mots(pred_dir / f"{seq.name}.txt", preds)

    ours = evaluate(dataset, pred_dir, [HOTA(), CLEAR(), Identity(), Count(), JAndF()]).combined

    # The oracle reads GT from the real data root but needs the sequence's
    # seqinfo-derived length; restrict it to the same single sequence.
    oracle = run_mots_challenge(
        root / "train", tmp_path / "trackers", {seq.name: seq.num_timesteps}
    )
    _assert_metrics_equal(ours, oracle, BOX_METRICS)
    for field, oracle_value in oracle["JAndF"].items():
        assert np.array_equal(
            np.asarray(ours["JAndF"][field], dtype=float),
            np.asarray(oracle_value, dtype=float),
            equal_nan=True,
        ), f"JAndF.{field}"


def test_perturbation_is_seeded_and_independently_numbered():
    from moteval.data.model import FrameConvention
    from moteval.formats.mot_txt import Track

    convention = FrameConvention(name="1-indexed", first_frame=1)
    gt = tuple(
        Track(frame=f, track_id=tid, x=100.0 * tid, y=50.0, w=20.0, h=40.0, conf=1.0)
        for f in range(1, 11)
        for tid in (3, 8)
    )
    a = perturb_box_tracks(gt, 10, convention, seed=7)
    b = perturb_box_tracks(gt, 10, convention, seed=7)
    c = perturb_box_tracks(gt, 10, convention, seed=8)
    assert a == b  # reproducible
    assert a != c  # seed actually matters
    gt_ids = {t.track_id for t in gt}
    assert not gt_ids & {t.track_id for t in a}  # independently numbered
