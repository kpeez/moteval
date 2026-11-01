"""Command-line interface for moteval.

The CLI is the Rust binary's (``src/cli.rs``): `main` runs it in-process
and writes its output to `sys.stdout` and `sys.stderr`, so the ``moteval`` console
script (``moteval._moteval:main``) and this function behave the same.
"""

import sys
from collections.abc import Sequence

from moteval import _moteval


def main(argv: Sequence[str] | None = None) -> int:
    """Run the moteval command-line interface on ``argv`` (default ``sys.argv[1:]``).

    Returns 0 on success. A usage error raises ``SystemExit(2)`` after the usage text, as
    argparse does; an error Python moteval did not catch (an I/O error, for example)
    raises ``SystemExit(1)``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    code, out, err = _moteval.cli_run(args)
    sys.stdout.write(out)
    sys.stderr.write(err)
    if code != 0:
        raise SystemExit(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
