"""Declarative benchmark downloads executed by one shared engine."""

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Literal

DEFAULT_DATA_ROOT = Path("data/benchmarks")
_CHIMPACT_REPOSITORY = "https://github.com/ShirleyMaxx/ChimpACT"
_MOTS20_URL = "https://motchallenge.net/data/MOTS.zip"
_PANAF500_URL = "https://data.bris.ac.uk/datasets/1h73erszj3ckn2qjwm4sqmr2wt/PanAf500"
_UAVDT_CHUNK_SIZE = 16 * 1024 * 1024
_UAVDT_MAX_STUBS = 20
_UAVDT_STUB_SNIFF_MAX = 65_536


@dataclass(frozen=True)
class LayoutCheck:
    """Relative directory markers that prove a loader can find its data."""

    markers: tuple[str, ...]
    description: str

    def is_present(self, benchmark_root: Path) -> bool:
        return all((benchmark_root / marker).is_dir() for marker in self.markers)


@dataclass(frozen=True)
class GDriveFolder:
    folder_id: str
    retry_delays: tuple[float, ...] = (300.0, 900.0, 2700.0)


@dataclass(frozen=True)
class HFSnapshot:
    repository: str
    revision: str


@dataclass(frozen=True)
class HttpFile:
    filename: str
    url: str
    expected_size: int | None


@dataclass(frozen=True)
class DriveFile:
    filename: str
    file_id: str
    expected_size: int


@dataclass(frozen=True)
class PanAf500Files:
    """Scrape and fetch the complete validated PanAf500 index."""


@dataclass(frozen=True)
class ZipExtraction:
    filename: str
    prefix_map: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SportsMotExtraction:
    archives: tuple[str, ...] = ("dataset/train.tar", "dataset/val.tar", "dataset/test.tar")


@dataclass(frozen=True)
class Mots20Extraction:
    filename: str = "MOTS.zip"


@dataclass(frozen=True)
class NormalizeFolder:
    marker: str


FetchStep = (
    GDriveFolder
    | HFSnapshot
    | HttpFile
    | DriveFile
    | PanAf500Files
    | ZipExtraction
    | SportsMotExtraction
    | Mots20Extraction
    | NormalizeFolder
)


@dataclass(frozen=True)
class DownloadSpec:
    """One benchmark's source declarations and loader-layout contract."""

    name: str
    fetch_steps: tuple[FetchStep, ...]
    expected_layout: LayoutCheck
    unfetchable_reason: str | None = None
    bundled: bool = False

    @property
    def fetchable(self) -> bool:
        return self.unfetchable_reason is None


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    availability: Literal["fetchable", "unfetchable", "bundled"]


@dataclass(frozen=True)
class BenchmarkState:
    name: str
    state: Literal["present", "absent", "bundled"]
    path: Path | None


_ANIMALTRACK_ID = "1P0oaPRruthyALztjJW8nbOegOpU_szew"
_BFT_ID = "140mPnOVZY-2apH76at9yYuVGIDWOvsH_"
_GMOT40_FILES = (
    HttpFile(
        filename="GenericMOT_JPEG_Sequence.zip",
        url="https://github.com/Spritea/GMOT40/releases/download/v0.1/GenericMOT_JPEG_Sequence.zip",
        expected_size=1_498_450_771,
    ),
    HttpFile(
        filename="track_label.zip",
        url="https://github.com/Spritea/GMOT40/releases/download/v0.1/track_label.zip",
        expected_size=1_978_567,
    ),
)
_UAVDT_FILES = (
    DriveFile(
        filename="UAV-benchmark-M.zip",
        file_id="1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc",
        expected_size=6_790_452_113,
    ),
    DriveFile(
        filename="UAV-benchmark-MOTD_v1.0.zip",
        file_id="19498uJd7T9w4quwnQEy62nibt3uyT9pq",
        expected_size=245_719_325,
    ),
)

_SPEC_DECLARATIONS = (
    DownloadSpec(
        name="animaltrack",
        fetch_steps=(GDriveFolder(_ANIMALTRACK_ID), NormalizeFolder("gt_all")),
        expected_layout=LayoutCheck(("gt_all",), "gt_all/<seq>_gt.txt"),
    ),
    DownloadSpec(
        name="bft",
        fetch_steps=(GDriveFolder(_BFT_ID), NormalizeFolder("annotations_mot")),
        expected_layout=LayoutCheck(("annotations_mot",), "annotations_mot/<split>/<seq>.txt"),
    ),
    DownloadSpec(
        name="chimpact",
        fetch_steps=(),
        expected_layout=LayoutCheck(
            ("ChimpACT_release_v1/labels",), "ChimpACT_release_v1/labels/<clip>.json"
        ),
        unfetchable_reason=(
            "ChimpACT has no stable programmatic artifact: the official repository only links "
            "a gated Google Form. Acquire it manually from "
            f"{_CHIMPACT_REPOSITORY} and place COCO JSON files at "
            "<root>/chimpact/ChimpACT_release_v1/labels/<clip>.json"
        ),
    ),
    DownloadSpec(
        name="dancetrack",
        fetch_steps=(
            HFSnapshot(
                repository="noahcao/dancetrack",
                revision="a0ba42ac690c41e9850a20e76a0b9450a6fb6a47",
            ),
            ZipExtraction("train1.zip", (("train1", "train"),)),
            ZipExtraction("train2.zip", (("train2", "train"),)),
            ZipExtraction("val.zip", (("val", "val"),)),
            ZipExtraction("test1.zip", (("test1", "test"),)),
            ZipExtraction("test2.zip", (("test2", "test"),)),
        ),
        expected_layout=LayoutCheck(("val",), "val/<seq>/{gt/gt.txt,seqinfo.ini}"),
    ),
    DownloadSpec(
        name="gmot40",
        fetch_steps=(*_GMOT40_FILES, ZipExtraction("track_label.zip")),
        expected_layout=LayoutCheck(("track_label",), "track_label/<seq>.txt"),
    ),
    DownloadSpec(
        name="mots20",
        fetch_steps=(HttpFile("MOTS.zip", _MOTS20_URL, None), Mots20Extraction()),
        expected_layout=LayoutCheck(("train",), "train/<seq>/{gt/gt.txt,seqinfo.ini}"),
    ),
    DownloadSpec(
        name="panaf500",
        fetch_steps=(PanAf500Files(),),
        expected_layout=LayoutCheck(
            ("annotations/validation",), "annotations/<split>/<video>.json"
        ),
    ),
    DownloadSpec(
        name="sportsmot",
        fetch_steps=(
            HFSnapshot(
                repository="MCG-NJU/SportsMOT",
                revision="1b0b418ea611a8e934f1e773313e719e4fa1265e",
            ),
            SportsMotExtraction(),
        ),
        expected_layout=LayoutCheck(("val",), "val/<seq>/{gt/gt.txt,seqinfo.ini}"),
    ),
    DownloadSpec(
        name="toy",
        fetch_steps=(),
        expected_layout=LayoutCheck((), "bundled in memory"),
        bundled=True,
    ),
    DownloadSpec(
        name="uavdt",
        fetch_steps=(*_UAVDT_FILES, ZipExtraction("UAV-benchmark-MOTD_v1.0.zip")),
        expected_layout=LayoutCheck(
            ("UAV-benchmark-MOTD_v1.0/GT",),
            "UAV-benchmark-MOTD_v1.0/GT/<seq>_gt.txt",
        ),
    ),
)
SPECS = {spec.name: spec for spec in _SPEC_DECLARATIONS}


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_row = False
        self._in_cell = False
        self._href = ""
        self._cell_text = ""
        self._cells: list[str] = []
        self.entries: list[tuple[str, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._in_row = True
            self._href = ""
            self._cells = []
        if not self._in_row:
            return
        if tag == "a":
            self._href = dict(attrs).get("href") or ""
        if tag == "td":
            self._in_cell = True
            self._cell_text = ""

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_cell:
            self._cells.append(self._cell_text.strip())
            self._in_cell = False
        if tag != "tr" or not self._in_row:
            return
        self._in_row = False
        filename = urllib.parse.unquote(self._href)
        if not filename or filename.startswith(".") or "/" in filename or not self._cells:
            return
        size = self._cells[-1]
        if filename.endswith((".mp4", ".json")) and size.isdigit():
            self.entries.append((filename, int(size)))


def benchmark_names() -> tuple[str, ...]:
    """Return every managed benchmark name in stable order."""
    return tuple(SPECS)


def list_benchmarks() -> tuple[BenchmarkInfo, ...]:
    """Return fetchability information for all managed benchmarks."""
    return tuple(
        BenchmarkInfo(
            name=spec.name,
            availability=(
                "bundled" if spec.bundled else "fetchable" if spec.fetchable else "unfetchable"
            ),
        )
        for spec in SPECS.values()
    )


def benchmark_status(root: Path) -> tuple[BenchmarkState, ...]:
    """Check loader-layout markers below a managed data root."""
    return tuple(
        BenchmarkState(
            name=spec.name,
            state=(
                "bundled"
                if spec.bundled
                else "present"
                if spec.expected_layout.is_present(root / spec.name)
                else "absent"
            ),
            path=None if spec.bundled else root / spec.name,
        )
        for spec in SPECS.values()
    )


def _unknown_benchmark(name: str) -> ValueError:
    return ValueError(f"unknown benchmark {name!r}; valid names: {', '.join(benchmark_names())}")


def _run_command(command: list[str]) -> None:
    try:
        result = subprocess.run(command, check=False)
    except OSError as error:
        raise RuntimeError(f"cannot run {command[0]!r}: {error}") from error
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed with exit code {result.returncode}")


def _fetch_gdrive_folder(step: GDriveFolder, destination: Path) -> None:
    url = f"https://drive.google.com/drive/folders/{step.folder_id}"
    errors: list[str] = []
    for attempt in range(len(step.retry_delays) + 1):
        try:
            _run_command(["gdown", "--folder", url, "-O", str(destination), "--continue"])
            return
        except RuntimeError as error:
            errors.append(str(error))
        if attempt < len(step.retry_delays):
            delay = step.retry_delays[attempt]
            print(f"gdown attempt {attempt + 1} failed; retrying in {delay:g} seconds")
            time.sleep(delay)
    raise RuntimeError(f"gdown failed after {len(errors)} attempts: {errors[-1]}")


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache/huggingface/token"
    try:
        token = token_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def _fetch_hf_snapshot(step: HFSnapshot, destination: Path) -> None:
    command = [
        "hf",
        "download",
        step.repository,
        "--repo-type",
        "dataset",
        "--revision",
        step.revision,
        "--local-dir",
        str(destination),
    ]
    token = _hf_token()
    if token is not None:
        command.extend(("--token", token))
    _run_command(command)


def _fetch_http(step: HttpFile, destination: Path) -> None:
    target = destination / step.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if step.expected_size is not None and target.is_file():
        size = target.stat().st_size
        if size == step.expected_size:
            return
        if size > step.expected_size:
            target.unlink()
    _run_command(["curl", "-L", "--fail", "-C", "-", "-o", str(target), step.url])
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"download produced no data: {target}")
    if step.expected_size is not None and target.stat().st_size != step.expected_size:
        raise RuntimeError(
            f"download size mismatch for {target}: {target.stat().st_size} != {step.expected_size}"
        )


def _fetch_drive_chunked(step: DriveFile, destination: Path) -> None:
    target = destination / step.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == step.expected_size:
        return
    if target.exists() and target.stat().st_size > step.expected_size:
        target.unlink()
    temporary = target.with_suffix(target.suffix + ".chunk")
    start = target.stat().st_size if target.exists() else 0
    url = (
        f"https://drive.usercontent.google.com/download?id={step.file_id}&export=download&confirm=t"
    )
    stubs = 0
    while start < step.expected_size:
        end = min(start + _UAVDT_CHUNK_SIZE, step.expected_size) - 1
        expected = end - start + 1
        try:
            _run_command(
                [
                    "curl",
                    "-sS",
                    "-L",
                    "--fail",
                    "-r",
                    f"{start}-{end}",
                    "-o",
                    str(temporary),
                    url,
                ]
            )
        except RuntimeError:
            temporary.unlink(missing_ok=True)
            stubs += 1
            if stubs >= _UAVDT_MAX_STUBS:
                raise RuntimeError(
                    f"aborted after {stubs} consecutive Drive quota stubs at offset {start}; "
                    "rerun after a quiet window to resume"
                ) from None
            continue
        received = temporary.stat().st_size if temporary.exists() else 0
        head = temporary.read_bytes()[:4] if 0 < received <= _UAVDT_STUB_SNIFF_MAX else b""
        if received != expected or head.startswith(b"<"):
            temporary.unlink(missing_ok=True)
            stubs += 1
            if stubs >= _UAVDT_MAX_STUBS:
                raise RuntimeError(
                    f"aborted after {stubs} consecutive Drive quota stubs at offset {start}; "
                    "rerun after a quiet window to resume"
                )
            continue
        stubs = 0
        with target.open("ab") as output, temporary.open("rb") as chunk:
            shutil.copyfileobj(chunk, output)
        temporary.unlink()
        start += expected
        print(f"{target.name}: {start}/{step.expected_size} ({start / step.expected_size:.1%})")
    if target.stat().st_size != step.expected_size:
        raise RuntimeError(f"download size mismatch for {target}")


def _safe_relative(name: str) -> Path:
    path = PurePosixPath(name)
    while path.parts and path.parts[0] == ".":
        path = PurePosixPath(*path.parts[1:])
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"unsafe archive member: {name!r}")
    return Path(*path.parts)


def _mapped_path(relative: Path, prefix_map: tuple[tuple[str, str], ...]) -> Path | None:
    if not prefix_map:
        return relative
    for source_raw, destination_raw in prefix_map:
        source = Path(source_raw)
        try:
            remainder = relative.relative_to(source)
        except ValueError:
            continue
        return Path(destination_raw) / remainder
    return None


def _file_crc(path: Path) -> int:
    checksum = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _find_archive(destination: Path, filename: str) -> Path:
    direct = destination / filename
    if direct.is_file():
        return direct
    matches = [path for path in destination.rglob(Path(filename).name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {filename!r} below {destination}; found {matches}"
        )
    return matches[0]


def _extract_zip(step: ZipExtraction, destination: Path) -> None:
    archive_path = _find_archive(destination, step.filename)
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = _safe_relative(member.filename)
            mapped = _mapped_path(relative, step.prefix_map)
            if mapped is None:
                continue
            target = destination / mapped
            if target.exists():
                if target.stat().st_size == member.file_size and _file_crc(target) == member.CRC:
                    extracted += 1
                    continue
                raise RuntimeError(f"archive extraction collision at {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
    if not extracted:
        raise RuntimeError(f"archive mapping matched no files: {archive_path}")


def _tar_member_matches(archive: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> bool:
    if target.stat().st_size != member.size:
        return False
    source = archive.extractfile(member)
    if source is None:
        return False
    with source, target.open("rb") as existing:
        while True:
            source_chunk = source.read(1024 * 1024)
            if source_chunk != existing.read(1024 * 1024):
                return False
            if not source_chunk:
                return True


def _extract_sportsmot(step: SportsMotExtraction, destination: Path) -> None:
    for archive_name in step.archives:
        split = Path(archive_name).stem
        source_prefix = Path("sportsmot_publish") / "dataset" / split
        archive_path = _find_archive(destination, archive_name)
        extracted = 0
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                relative = _safe_relative(member.name)
                try:
                    remainder = relative.relative_to(source_prefix)
                except ValueError:
                    continue
                target = destination / split / remainder
                if target.exists():
                    if _tar_member_matches(archive, member, target):
                        extracted += 1
                        continue
                    raise RuntimeError(f"archive extraction collision at {target}")
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read tar member {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
        if not extracted:
            raise RuntimeError(
                f"archive {archive_path} has no files below {source_prefix.as_posix()}"
            )


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(1024 * 1024)
            if left_chunk != right_file.read(1024 * 1024):
                return False
            if not left_chunk:
                return True


def _merge_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            if _files_equal(path, target):
                continue
            raise RuntimeError(f"layout flattening collision at {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _extract_mots20(step: Mots20Extraction, destination: Path) -> None:
    archive_path = _find_archive(destination, step.filename)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".moteval-mots20-") as temporary:
        extracted_root = Path(temporary)
        _extract_zip(ZipExtraction(str(archive_path)), extracted_root)
        for split in ("train", "test"):
            candidates = [
                path
                for path in extracted_root.rglob(split)
                if path.is_dir()
                and any((sequence / "seqinfo.ini").is_file() for sequence in path.iterdir())
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected exactly one MOTS20 {split!r} tree in {archive_path}; "
                    f"found {candidates}"
                )
            _merge_tree(candidates[0], destination / split)


def _normalize_folder(step: NormalizeFolder, destination: Path) -> None:
    if (destination / step.marker).is_dir():
        return
    matches = [path for path in destination.rglob(Path(step.marker).name) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected layout marker {step.marker!r} below {destination}; found {matches}"
        )
    container = matches[0].parent
    if container == destination:
        return
    _merge_tree(container, destination)


def _list_panaf_files() -> list[tuple[str, int]]:
    indexes = (
        ("videos", 500, ".mp4"),
        ("annotations/train", 400, ".json"),
        ("annotations/validation", 25, ".json"),
        ("annotations/test", 75, ".json"),
    )
    files: list[tuple[str, int]] = []
    for relative_dir, expected_count, suffix in indexes:
        with urllib.request.urlopen(f"{_PANAF500_URL}/{relative_dir}/", timeout=60) as response:
            html = response.read().decode("utf-8")
        parser = _IndexParser()
        parser.feed(html)
        entries = sorted(
            (
                (f"{relative_dir}/{filename}", size)
                for filename, size in parser.entries
                if filename.endswith(suffix)
            ),
            key=lambda entry: entry[0],
        )
        if len(entries) != expected_count:
            raise RuntimeError(
                f"{relative_dir} index has {len(entries)} {suffix} files; expected {expected_count}"
            )
        files.extend(entries)
    if len({path for path, _size in files}) != len(files):
        raise RuntimeError("PanAf500 index contains duplicate paths")
    return sorted(files)


def _fetch_panaf500(destination: Path) -> None:
    files = _list_panaf_files()
    manifest = [{"path": path, "size": size} for path, size in files]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for index, (relative_path, expected_size) in enumerate(files, 1):
        encoded_path = urllib.parse.quote(relative_path, safe="/")
        print(f"[{index}/{len(files)}] {relative_path}")
        _fetch_http(
            HttpFile(relative_path, f"{_PANAF500_URL}/{encoded_path}", expected_size),
            destination,
        )


def _execute_step(step: FetchStep, destination: Path) -> None:
    if isinstance(step, GDriveFolder):
        _fetch_gdrive_folder(step, destination)
    elif isinstance(step, HFSnapshot):
        _fetch_hf_snapshot(step, destination)
    elif isinstance(step, HttpFile):
        _fetch_http(step, destination)
    elif isinstance(step, DriveFile):
        _fetch_drive_chunked(step, destination)
    elif isinstance(step, PanAf500Files):
        _fetch_panaf500(destination)
    elif isinstance(step, ZipExtraction):
        _extract_zip(step, destination)
    elif isinstance(step, SportsMotExtraction):
        _extract_sportsmot(step, destination)
    elif isinstance(step, Mots20Extraction):
        _extract_mots20(step, destination)
    elif isinstance(step, NormalizeFolder):
        _normalize_folder(step, destination)
    else:
        raise AssertionError(f"unhandled fetch step: {step!r}")


def download_benchmark(name: str, root: Path) -> Path:
    """Fetch one benchmark and verify the exact loader-visible layout."""
    try:
        spec = SPECS[name]
    except KeyError:
        raise _unknown_benchmark(name) from None
    if spec.unfetchable_reason is not None:
        raise RuntimeError(spec.unfetchable_reason)
    if spec.bundled:
        return root
    destination = root / spec.name
    destination.mkdir(parents=True, exist_ok=True)
    for step in spec.fetch_steps:
        _execute_step(step, destination)
    if not spec.expected_layout.is_present(destination):
        raise RuntimeError(
            f"download completed but {spec.name} layout is invalid at {destination}; "
            f"expected {spec.expected_layout.description}"
        )
    return destination
