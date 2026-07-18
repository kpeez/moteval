"""Command-line interface for moteval."""

import argparse
import csv
import inspect
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from moteval import CLEAR, HOTA, Count, Identity, JAndF, Metric, MOTDataset, TrackMAP, evaluate
from moteval.benchmarks.base import DATASETS
from moteval.download import (
    DEFAULT_DATA_ROOT,
    benchmark_status,
    download_benchmark,
    list_benchmarks,
)
from moteval.results import EvaluationResult, iter_csv_rows, to_json_dict

_DEFAULT_METRICS = "hota,clear,identity,count"
_METRICS: dict[str, type[Metric]] = {
    "hota": HOTA,
    "clear": CLEAR,
    "identity": Identity,
    "count": Count,
    "track_map": TrackMAP,
    "jf": JAndF,
}
_HEADLINES = (
    ("HOTA", "HOTA", "HOTA"),
    ("HOTA", "DetA", "DetA"),
    ("HOTA", "AssA", "AssA"),
    ("CLEAR", "MOTA", "MOTA"),
    ("CLEAR", "MOTP", "MOTP"),
    ("CLEAR", "IDSW", "IDSW"),
    ("Identity", "IDF1", "IDF1"),
    ("Count", "Dets", "Dets"),
    ("Count", "GT_Dets", "GT_Dets"),
)
_INTEGER_HEADLINES = {"IDSW", "Dets", "GT_Dets"}
_INTEGER_FORMAT = "{:d}"
_RATIO_FORMAT = "{:1.5g}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moteval")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="evaluate prediction files")
    run.add_argument("--dataset", required=True, help="registered dataset name")
    run.add_argument("--split", help="dataset split (uses the loader default when omitted)")
    run.add_argument("--gt", type=Path, help="ground-truth dataset root")
    run.add_argument("--pred", type=Path, required=True, help="prediction directory")
    run.add_argument("--metrics", default=_DEFAULT_METRICS, help="comma-separated metrics")
    run.add_argument("--out-csv", type=Path, help="write long-form CSV results")
    run.add_argument("--out-json", type=Path, help="write JSON results")
    run.set_defaults(handler=_run)

    data = subcommands.add_parser("data", help="manage benchmark ground truth")
    data.add_argument("--root", dest="data_root", type=Path, help="managed benchmark root")
    data_commands = data.add_subparsers(dest="data_command", required=True)

    data_list = data_commands.add_parser("list", help="list managed benchmarks")
    _add_data_root_override(data_list)
    data_list.set_defaults(handler=_data_list)

    data_status = data_commands.add_parser("status", help="check benchmark data layouts")
    _add_data_root_override(data_status)
    data_status.set_defaults(handler=_data_status)

    data_download = data_commands.add_parser("download", help="download benchmark data")
    data_download.add_argument("benchmark", help="benchmark name")
    _add_data_root_override(data_download)
    data_download.set_defaults(handler=_data_download)
    return parser


def _add_data_root_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        dest="subcommand_data_root",
        type=Path,
        help="managed benchmark root",
    )


def _data_root(args: argparse.Namespace) -> Path:
    if args.subcommand_data_root is not None:
        return args.subcommand_data_root
    if args.data_root is not None:
        return args.data_root
    if "MOTEVAL_DATA_ROOT" not in os.environ:
        return DEFAULT_DATA_ROOT
    raw_root = os.environ["MOTEVAL_DATA_ROOT"]
    if not raw_root.strip():
        raise ValueError("MOTEVAL_DATA_ROOT must not be empty")
    return Path(raw_root)


def _data_list(args: argparse.Namespace) -> int:
    _data_root(args)
    print("benchmark     availability")
    for benchmark in list_benchmarks():
        print(f"{benchmark.name:<13} {benchmark.availability}")
    return 0


def _data_status(args: argparse.Namespace) -> int:
    root = _data_root(args)
    print("benchmark     status    path")
    for benchmark in benchmark_status(root):
        path = "-" if benchmark.path is None else str(benchmark.path)
        print(f"{benchmark.name:<13} {benchmark.state:<9} {path}")
    return 0


def _data_download(args: argparse.Namespace) -> int:
    root = _data_root(args)
    destination = download_benchmark(args.benchmark, root)
    if args.benchmark == "toy":
        print("toy is bundled; no download required")
    else:
        print(f"downloaded {args.benchmark} to {destination}")
    return 0


def _resolve_metrics(raw_names: str) -> list[Metric]:
    names = [name.strip().lower() for name in raw_names.split(",") if name.strip()]
    unknown = sorted(set(names) - _METRICS.keys())
    if unknown:
        available = ", ".join(_METRICS)
        raise ValueError(f"unknown metric(s): {', '.join(unknown)}; available: {available}")
    if not names:
        available = ", ".join(_METRICS)
        raise ValueError(f"no metrics selected; available: {available}")
    return [_METRICS[name]() for name in names]


def _load_dataset(name: str, root: Path | None, split: str | None) -> MOTDataset:
    try:
        registered_loader = DATASETS.get(name)
    except KeyError as error:
        raise ValueError(error.args[0]) from None

    loader = cast(Callable[..., MOTDataset], registered_loader)
    parameters = inspect.signature(loader).parameters
    kwargs: dict[str, object] = {}
    if root is not None and "root" in parameters:
        kwargs["root"] = root
    if split is not None and "split" in parameters:
        kwargs["split"] = split
    return loader(**kwargs)


def _scalar(value: float | np.ndarray) -> float:
    if isinstance(value, np.ndarray):
        return float(np.mean(value))
    return float(value)


def _format_headline(value: float, label: str) -> str:
    if label in _INTEGER_HEADLINES:
        return _INTEGER_FORMAT.format(int(value))
    return _RATIO_FORMAT.format(100 * value)


def _table(result: EvaluationResult) -> str:
    columns = [column for column in _HEADLINES if column[0] in result.combined]
    headers = ["seq", *(label for _metric, _field, label in columns)]
    rows: list[list[str]] = []
    all_scores = (*result.per_sequence.items(), ("COMBINED", result.combined))
    for sequence, scores in all_scores:
        values = [sequence]
        for metric, field, label in columns:
            value = _scalar(scores[metric][field])
            values.append(_format_headline(value, label))
        rows.append(values)

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    template = "  ".join(
        f"{{:{'<' if index == 0 else '>'}{width}}}" for index, width in enumerate(widths)
    )
    return "\n".join([template.format(*headers), *(template.format(*row) for row in rows)])


def _write_csv(path: Path, result: EvaluationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("seq", "metric", "field", "value"))
        writer.writerows(iter_csv_rows(result))


def _write_json(path: Path, result: EvaluationResult, dataset: MOTDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(result, dataset=dataset.name, split=dataset.split)))


def _run(args: argparse.Namespace) -> int:
    metrics = _resolve_metrics(args.metrics)
    if args.gt is not None and not args.gt.is_dir():
        raise ValueError(f"ground-truth root not found or not a directory: {args.gt}")
    if not args.pred.is_dir():
        raise ValueError(f"prediction directory not found or not a directory: {args.pred}")

    dataset = _load_dataset(args.dataset, args.gt, args.split)
    result = evaluate(dataset, args.pred, metrics)
    print(_table(result))
    if args.out_csv is not None:
        _write_csv(args.out_csv, result)
    if args.out_json is not None:
        _write_json(args.out_json, result, dataset)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the moteval command-line interface."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
