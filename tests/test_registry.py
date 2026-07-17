import pytest

from moteval.benchmarks.base import DATASETS, load_dataset


def test_toy_is_registered():
    import moteval  # noqa: F401  (import triggers toy registration)

    assert "toy" in DATASETS.names()


def test_dancetrack_and_sportsmot_are_registered():
    import moteval  # noqa: F401  (import triggers benchmark registration)

    assert "dancetrack" in DATASETS.names()
    assert "sportsmot" in DATASETS.names()


def test_load_dataset_unknown_name_raises():
    with pytest.raises(KeyError):
        load_dataset("does-not-exist")
