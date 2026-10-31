"""Set up a moteval checkout or worktree: git hooks and the gitignored local links.

- git hooks: ``prek install`` in the main checkout only. A worktree shares ``.git/hooks`` with
  the main checkout, so installing there would point the shared hook at the worktree's venv.

- ``artifacts`` -> ``$MOTEVAL_ARTIFACTS_DIR`` (default ``~/artifacts/moteval``): run output
  (logs, metrics, profiles, benchmark results), shared by every worktree.
- ``.agents/docs`` -> ``<llmOS vault>/projects/moteval``: the project's notes and ADRs. The
  vault root is ``$LLMOS_ROOT``, else ``vault_root`` in ``~/.config/llmos/config.json``;
  without either, or without a ``vault_root`` key, this link is skipped.

Idempotent: an existing path is left as it is, and a link to another target is reported.
Run from the repository root (``mise run install`` does).
"""

import json
import os
import subprocess
from pathlib import Path


def vault_root() -> Path | None:
    if root := os.environ.get("LLMOS_ROOT"):
        return Path(root)
    config = Path.home() / ".config" / "llmos" / "config.json"
    if config.is_file() and (root := json.loads(config.read_text()).get("vault_root")):
        return Path(root)
    return None


def install_hooks() -> None:
    def git(flag: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", flag], check=True, capture_output=True, text=True
        ).stdout.strip()

    if Path(git("--git-dir")).resolve() != Path(git("--git-common-dir")).resolve():
        print("skipping hook install in a worktree")
        return
    subprocess.run(["uv", "run", "prek", "install"], check=True)


def link(path: Path, target: Path) -> None:
    if path.is_symlink():
        current = Path(os.readlink(path))
        status = (
            "ok" if current == target else f"left as is (points to {current}, expected {target})"
        )
        print(f"{path}: {status}")
        return
    if path.exists():
        print(f"{path}: left as is (a real directory or file, not a link)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    print(f"{path}: linked to {target}")


def main() -> None:
    install_hooks()

    artifacts = (
        Path(os.environ.get("MOTEVAL_ARTIFACTS_DIR") or Path.home() / "artifacts" / "moteval")
        .expanduser()
        .resolve()
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    link(Path("artifacts"), artifacts)

    root = vault_root()
    docs = (root.expanduser().resolve() / "projects" / "moteval") if root else None
    if docs is None or not docs.is_dir():
        print(".agents/docs: skipped (no llmOS vault with projects/moteval on this host)")
        return
    link(Path(".agents/docs"), docs)


if __name__ == "__main__":
    main()
