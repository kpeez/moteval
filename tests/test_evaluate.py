import pytest

from moteval import evaluate, load_dataset
from moteval.formats.mot_txt import write_mot
from moteval.metrics.count import Count


def _write_predictions_matching_gt(dataset, pred_dir):
    for seq in dataset.sequences:
        write_mot(pred_dir / f"{seq.name}.txt", list(seq.tracks))


def test_evaluate_returns_per_sequence_and_combined_count(tmp_path):
    dataset = load_dataset("toy")
    _write_predictions_matching_gt(dataset, tmp_path)

    result = evaluate(dataset, tmp_path, [Count()])

    # Each toy sequence: 2 ids over 5 frames = 10 dets.
    for seq in dataset.sequences:
        scores = result.per_sequence[seq.name]["Count"]
        assert scores == {"Dets": 10.0, "GT_Dets": 10.0, "IDs": 2.0, "GT_IDs": 2.0}

    assert result.combined["Count"] == {
        "Dets": 20.0,
        "GT_Dets": 20.0,
        "IDs": 4.0,
        "GT_IDs": 4.0,
    }


def test_evaluate_with_missing_prediction_file_reports_zero_preds(tmp_path):
    dataset = load_dataset("toy")
    # No prediction files written at all.
    result = evaluate(dataset, tmp_path, [Count()])
    assert result.combined["Count"]["Dets"] == 0.0
    assert result.combined["Count"]["GT_Dets"] == 20.0


def test_evaluate_rejects_duplicate_metric_classes(tmp_path):
    dataset = load_dataset("toy")
    with pytest.raises(ValueError) as exc:
        evaluate(dataset, tmp_path, [Count(), Count()])
    assert "Count" in str(exc.value)


def test_evaluate_rejects_multi_class_protocol(tmp_path):
    from dataclasses import replace

    toy = load_dataset("toy")
    multi = replace(toy, protocol=replace(toy.protocol, eval_classes=(1, 2)))
    with pytest.raises(ValueError, match="single-class"):
        evaluate(multi, tmp_path, [Count()])
