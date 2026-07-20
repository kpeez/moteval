install:
    uv sync --locked
    test -f .pre-commit-config.yaml && prek install || echo "no prek config; skipping hook install"

check:
    uv run ruff format .
    uv run ruff check --fix .
    uv run ty check

test:
    uv run pytest

test-real:
    uv run pytest -m real_data
