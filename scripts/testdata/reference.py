import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_TAG = "py-reference"
REFERENCE_PYTHON = REPO_ROOT / "artifacts" / "rust-port" / "ref-venv" / "bin" / "python"


def require_reference() -> None:
    if importlib.util.find_spec("moteval.data.convert") is None:
        raise SystemExit(
            f"{Path(sys.argv[0]).name} needs the pure-Python reference moteval "
            f"({REFERENCE_TAG}); run it with {REFERENCE_PYTHON}"
        )
