import sys
from collections.abc import Sequence

from moteval import _moteval


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    code, out, err = _moteval.cli_run(args)
    sys.stdout.write(out)
    sys.stderr.write(err)
    if code != 0:
        raise SystemExit(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
