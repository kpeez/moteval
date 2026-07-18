"""Benchmark registry seed: loaders keyed by name produce a `MOTDataset`."""

from collections.abc import Callable

from moteval.benchmarks.registry import Registry
from moteval.data.model import MOTDataset

DatasetLoader = Callable[[], MOTDataset]

DATASETS: Registry[DatasetLoader] = Registry("dataset")


def register_dataset(name: str) -> Callable[[DatasetLoader], DatasetLoader]:
    return DATASETS.register(name)


def load_dataset(name: str) -> MOTDataset:
    return DATASETS.get(name)()
