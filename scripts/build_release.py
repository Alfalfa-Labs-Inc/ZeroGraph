"""Build and independently rebuild canonical Independent release archives."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> None:
    candidate = PurePosixPath(name)
    if not name or "\x00" in name or "\\" in name or candidate.is_absolute():
        raise ValueError(f"unsafe archive member: {name!r}")
    if ".." in candidate.parts:
        raise ValueError(f"unsafe archive member: {name!r}")


def _reject_source_symlinks(source: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"release source contains a symbolic link: {path}")


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in sorted(source.iterdir(), key=lambda row: row.name):
        if path.name in {"__pycache__", ".pytest_cache", "build"} or path.name.endswith(
            (".egg-info", ".pyc", ".pyo")
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"release source contains a symbolic link: {path}")
        target = destination / path.name
        if path.is_dir():
            _copy_tree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)
        else:
            raise ValueError(f"unsupported release source member: {path}")


def _stage_source(source: Path, destination: Path) -> Path:
    staged = destination / "source"
    staged.mkdir(parents=True)
    for name in ("README.md", "pyproject.toml", "MANIFEST.in"):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release source lacks a regular {name}")
        shutil.copy2(path, staged / name)
    for name in ("scripts", "src", "tests"):
        path = source / name
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"release source lacks a regular {name}/ tree")
        _copy_tree(path, staged / name)
    return staged


def _normalize_sdist(source: Path, destination: Path, *, epoch: int) -> None:
    with (
        tarfile.open(source, "r:gz") as incoming,
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as output,
    ):
        for original in sorted(incoming.getmembers(), key=lambda row: row.name):
            _safe_name(original.name)
            if original.issym() or original.islnk() or original.isdev():
                raise ValueError(f"unsupported sdist member: {original.name!r}")
            member = copy.copy(original)
            member.mtime = epoch
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.pax_headers = {}
            member.mode = 0o755 if original.isdir() else 0o644
            payload = incoming.extractfile(original) if original.isfile() else None
            output.addfile(member, payload)


def _normalize_wheel(source: Path, destination: Path, *, epoch: int) -> None:
    timestamp = time.gmtime(max(epoch, 315532800))[:6]
    with (
        zipfile.ZipFile(source) as incoming,
        zipfile.ZipFile(
            destination,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as output,
    ):
        entries = incoming.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate members")
        for original in sorted(entries, key=lambda row: row.filename):
            _safe_name(original.filename)
            member = zipfile.ZipInfo(original.filename, date_time=timestamp)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = (0o100644 & 0xFFFF) << 16
            member.flag_bits = 0
            member.extra = b""
            member.comment = b""
            output.writestr(member, incoming.read(original))


def _raw_build(source: Path, python: Path, destination: Path, epoch: int) -> None:
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = str(epoch)
    environment["TMPDIR"] = str(destination.parent)
    completed = subprocess.run(
        [
            str(python),
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(destination),
            str(source),
        ],
        cwd=source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated archive build failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _canonical_build(source: Path, python: Path, destination: Path, epoch: int) -> dict:
    raw = destination / "raw"
    canonical = destination / "canonical"
    raw.mkdir(parents=True)
    canonical.mkdir()
    _raw_build(source, python, raw, epoch)
    wheels = list(raw.glob("*.whl"))
    sdists = list(raw.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("build must produce exactly one wheel and one sdist")
    wheel = canonical / wheels[0].name
    sdist = canonical / sdists[0].name
    _normalize_wheel(wheels[0], wheel, epoch=epoch)
    _normalize_sdist(sdists[0], sdist, epoch=epoch)
    return {
        wheel.name: {"path": wheel, "sha256": _sha256(wheel)},
        sdist.name: {"path": sdist, "sha256": _sha256(sdist)},
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-date-epoch", type=int, default=1787932800)
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    # Do not resolve a virtual-environment interpreter symlink to the base
    # executable: doing so discards the environment's build dependencies.
    python = Path(args.python).absolute()
    if output.exists():
        raise FileExistsError("release output already exists; archive it first")
    if args.source_date_epoch < 0:
        raise ValueError("source date epoch must be nonnegative")
    _reject_source_symlinks(source)
    with tempfile.TemporaryDirectory(
        prefix="independent-release-", dir="/dev/shm"
    ) as temporary:
        temporary_root = Path(temporary)
        staged = _stage_source(source, temporary_root / "staging")
        first = _canonical_build(
            staged, python, temporary_root / "first", args.source_date_epoch
        )
        second = _canonical_build(
            staged, python, temporary_root / "second", args.source_date_epoch
        )
        first_hashes = {name: row["sha256"] for name, row in first.items()}
        second_hashes = {name: row["sha256"] for name, row in second.items()}
        if first_hashes != second_hashes:
            raise RuntimeError("canonical rebuild did not reproduce artifact bytes")
        output.mkdir(parents=True)
        for name, row in first.items():
            (output / name).write_bytes(row["path"].read_bytes())
    checksums = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(first_hashes.items())
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="ascii")
    report = {
        "format_version": 1,
        "status": "passed",
        "package": "diffusionblocks-independent",
        "source_date_epoch": args.source_date_epoch,
        "python": sys.version,
        "artifacts": first_hashes,
        "independent_rebuild_byte_identical": True,
        "claim_scope": "canonical archives under this source and pinned build environment",
    }
    (output / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
