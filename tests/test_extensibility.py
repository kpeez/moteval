import json

import pytest

import moteval


def test_custom_dataset_evaluates_through_public_api(tmp_path):
    annotations_path = tmp_path / "custom-annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {"frame": 1, "id": 1, "box": [0, 0, 10, 10], "class_id": 1},
                {"frame": 1, "id": 2, "box": [500, 500, 10, 10], "class_id": 2},
            ]
        )
    )
    convention = moteval.FrameConvention(name="test-ext-1-indexed", first_frame=1)
    distractor_protocol = moteval.Protocol(
        name="test-ext-distractors",
        frame_convention=convention,
        eval_classes=(1,),
        distractor_classes=(2,),
    )

    @moteval.register_dataset("test-ext-json-distractors")
    def load_custom_dataset() -> moteval.MOTDataset:
        rows = json.loads(annotations_path.read_text())
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
        sequence = moteval.GtSequence(name="custom-sequence", num_timesteps=1, tracks=tracks)
        return moteval.MOTDataset(
            name="test-ext-json-distractors",
            split="test",
            sequences=(sequence,),
            protocol=distractor_protocol,
        )

    dataset = moteval.load_dataset("test-ext-json-distractors")
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "custom-sequence.txt").write_text(
        "1,101,0,0,10,10,1\n1,202,500,500,10,10,1\n"
    )

    with_distractor = moteval.evaluate(
        dataset,
        predictions_dir,
        [moteval.HOTA(), moteval.CLEAR(), moteval.Identity(), moteval.Count()],
    ).combined

    assert with_distractor["HOTA"]["HOTA(0)"] == 1.0
    assert with_distractor["CLEAR"]["MOTA"] == 1.0
    assert with_distractor["CLEAR"]["CLR_FP"] == 0.0
    assert with_distractor["Identity"] == {
        "IDF1": 1.0,
        "IDR": 1.0,
        "IDP": 1.0,
        "IDTP": 1.0,
        "IDFN": 0.0,
        "IDFP": 0.0,
    }
    assert with_distractor["Count"] == {
        "Dets": 1.0,
        "GT_Dets": 1.0,
        "IDs": 1.0,
        "GT_IDs": 1.0,
    }

    no_distractor_dataset = moteval.MOTDataset(
        name=dataset.name,
        split=dataset.split,
        sequences=dataset.sequences,
        protocol=moteval.Protocol(
            name="test-ext-no-distractors",
            frame_convention=convention,
            eval_classes=(1,),
        ),
    )
    without_distractor = moteval.evaluate(
        no_distractor_dataset,
        predictions_dir,
        [moteval.HOTA(), moteval.CLEAR(), moteval.Identity(), moteval.Count()],
    ).combined

    assert without_distractor["HOTA"]["HOTA(0)"] == 2**-0.5
    assert without_distractor["CLEAR"]["MOTA"] == 0.0
    assert without_distractor["CLEAR"]["CLR_FP"] == 1.0
    assert without_distractor["Identity"]["IDFP"] == 1.0
    assert without_distractor["Identity"]["IDF1"] == 2 / 3
    assert without_distractor["Count"] == {
        "Dets": 2.0,
        "GT_Dets": 1.0,
        "IDs": 2.0,
        "GT_IDs": 1.0,
    }


def test_duplicate_dataset_registration_names_the_dataset():
    name = "test-ext-duplicate-registration"

    def loader() -> moteval.MOTDataset:
        raise AssertionError("registration must not call the loader")

    moteval.register_dataset(name)(loader)
    with pytest.raises(ValueError, match=name):
        moteval.register_dataset(name)(loader)


def test_unknown_dataset_lists_registered_names():
    with pytest.raises(KeyError) as exc_info:
        moteval.load_dataset("nonexistent")

    message = str(exc_info.value)
    assert "registered:" in message
    assert "toy" in message
