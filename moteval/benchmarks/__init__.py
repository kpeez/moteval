"""Built-in benchmark loaders, keyed by name.

`BENCHMARKS` is the explicit index the CLI resolves ``--dataset`` names against;
`load_dataset` is the public by-name entry point. Custom data never registers
anything: standard-layout directories load through `load_motchallenge` /
`load_mots` (see `motchallenge.py` / `mots20.py`), and any other source format
just constructs a `MOTDataset` directly and calls `evaluate`.
"""

from collections.abc import Callable
from pathlib import Path

from moteval.benchmarks.animaltrack import load_animaltrack
from moteval.benchmarks.bft import load_bft
from moteval.benchmarks.chimpact import load_chimpact
from moteval.benchmarks.dancetrack import load_dancetrack
from moteval.benchmarks.gmot40 import load_gmot40
from moteval.benchmarks.mots20 import load_mots20
from moteval.benchmarks.panaf500 import load_panaf500
from moteval.benchmarks.sportsmot import load_sportsmot
from moteval.benchmarks.uavdt import load_uavdt
from moteval.data.model import GtSequence, MaskGtSequence, MOTDataset

# Every loader accepts (root=None, split=<benchmark default>) keywords.
DatasetLoader = Callable[..., MOTDataset[GtSequence | MaskGtSequence]]

BENCHMARKS: dict[str, DatasetLoader] = {
    "animaltrack": load_animaltrack,
    "bft": load_bft,
    "chimpact": load_chimpact,
    "dancetrack": load_dancetrack,
    "gmot40": load_gmot40,
    "mots20": load_mots20,
    "panaf500": load_panaf500,
    "sportsmot": load_sportsmot,
    "uavdt": load_uavdt,
}


def load_dataset(
    name: str, root: str | Path | None = None, split: str | None = None
) -> MOTDataset[GtSequence | MaskGtSequence]:
    """Load a built-in benchmark by name, optionally overriding root and split.

    ``root``/``split`` fall back to the benchmark's defaults when omitted.
    """
    if name not in BENCHMARKS:
        known = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"unknown dataset {name!r}; available: {known}")
    loader = BENCHMARKS[name]
    if split is None:
        return loader(root=root)
    return loader(root=root, split=split)
