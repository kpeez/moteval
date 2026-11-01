"""The pure-Python reference moteval that the dump scripts take their expected values from.

The ``moteval`` package in this checkout is a thin wrapper over the Rust extension, so
expected values computed with it would test Rust against itself. The dump scripts run
under the reference venv instead, where ``moteval`` is the ``py-reference`` tag installed
as a package (``artifacts/rust-port/ref-venv``; ``rust/README.md`` shows how to build it).
Each script puts the repository root at the END of ``sys.path``: ``moteval`` then
resolves to the installed reference, while ``tests``, ``scripts``, and ``rust.scripts``
resolve to this checkout.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_TAG = "py-reference"
REFERENCE_PYTHON = REPO_ROOT / "artifacts" / "rust-port" / "ref-venv" / "bin" / "python"


def require_reference() -> None:
    """Exit unless ``import moteval`` gives the pure-Python reference."""
    if importlib.util.find_spec("moteval.data.convert") is None:
        raise SystemExit(
            f"{Path(sys.argv[0]).name} needs the pure-Python reference moteval "
            f"({REFERENCE_TAG}); run it with {REFERENCE_PYTHON}"
        )
