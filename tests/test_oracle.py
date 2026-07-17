import numpy as np
import pytest

from tests.oracle.runner import run_mot_challenge


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(",".join(str(v) for v in row) for row in rows) + "\n")


def _build_perfect_sequence(root):
    seq = "SEQ01"
    # MOT gt: frame,id,x,y,w,h,zero_marked,class(pedestrian=1),visibility
    gt = [
        [1, 1, 10, 10, 20, 40, 1, 1, 1],
        [2, 1, 12, 10, 20, 40, 1, 1, 1],
        [1, 2, 100, 50, 30, 60, 1, 1, 1],
        [2, 2, 102, 50, 30, 60, 1, 1, 1],
    ]
    # tracker: frame,id,x,y,w,h,conf (identical boxes/ids => perfect tracking)
    pred = [
        [1, 1, 10, 10, 20, 40, 1],
        [2, 1, 12, 10, 20, 40, 1],
        [1, 2, 100, 50, 30, 60, 1],
        [2, 2, 102, 50, 30, 60, 1],
    ]
    _write(root / "gt" / seq / "gt" / "gt.txt", gt)
    _write(root / "trackers" / "oracle" / "data" / f"{seq}.txt", pred)
    return {seq: 2}


def test_oracle_evaluates_perfect_mot_sequence(tmp_path):
    seq_lengths = _build_perfect_sequence(tmp_path)

    res = run_mot_challenge(tmp_path / "gt", tmp_path / "trackers", seq_lengths)

    assert set(res) == {"HOTA", "CLEAR", "Identity", "Count"}

    # Perfect tracking: every reported summary field is exact.
    assert res["CLEAR"]["MOTA"] == pytest.approx(1.0)
    assert res["CLEAR"]["MOTP"] == pytest.approx(1.0)
    assert res["CLEAR"]["IDSW"] == 0
    assert np.mean(res["HOTA"]["HOTA"]) == pytest.approx(1.0)
    assert np.mean(res["HOTA"]["DetA"]) == pytest.approx(1.0)
    assert np.mean(res["HOTA"]["AssA"]) == pytest.approx(1.0)
    assert res["Identity"]["IDF1"] == pytest.approx(1.0)
    assert res["Count"] == {"Dets": 4, "GT_Dets": 4, "IDs": 2, "GT_IDs": 2}


def test_motmetrics_imports():
    import motmetrics

    assert motmetrics.__version__
