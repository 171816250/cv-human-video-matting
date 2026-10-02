"""Create, verify, extract and back up a portable GPU-server deployment.

Only the standard library is needed to inspect an archive on a new server.
Archives contain no virtual environment, credentials or server-specific state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo


CHUNK = 1024 * 1024
MODEL_DEST = "weights/portable-rvm"
SOURCE_FILES = (
    ".gitignore", ".gitattributes", "app.py", "pyproject.toml", "LICENSE", "README.md",
    "CHANGELOG.md", "CONTRIBUTING.md", "PROJECT_COMPLETE.md", "SUPERPOWERS.md",
    "Start-Demo.cmd", "Start-Realtime.cmd", "Connect-Remote-Realtime.ps1",
    "requirements.txt", "requirements-demo.txt", "requirements-realtime.txt",
    "requirements-server-realtime.txt", "requirements-optional.txt", "requirements-dev.txt",
    "requirements-dlf-cu117.txt", "requirements-lock-windows-py310-cpu.txt",
    "requirements-lock-windows-py310-cu121.txt",
)
SOURCE_DIRS = ("src", "configs", "scripts", "docs", "tests", ".github")
BACKUP_DIRS = ("data/input", "data/backgrounds", "data/output", "logs", "weights")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class PortableError(ValueError):
    """A transfer would be incomplete, unsafe or overwrite existing data."""


def _relative(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise PortableError(f"Unsafe archive path: {name!r}")
    return name


def _file_info(stream) -> dict:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
        size += len(block)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _hash_path(path: Path) -> dict:
    with path.open("rb") as stream:
        return _file_info(stream)


def _reject_link(path: Path) -> None:
    info = path.lstat()
    reparse = getattr(info, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400,
    )
    if stat.S_ISLNK(info.st_mode) or reparse:
        raise PortableError(f"Linked files are not portable: {path}")


def _walk(directory: Path, prefix: str, *, skip_links: frozenset[str] = frozenset()) -> dict[str, Path]:
    """Include ordinary files; never follow links or carry build/cache files."""
    files = {}
    if not directory.exists():
        return files
    _reject_link(directory)
    if not directory.is_dir():
        raise PortableError(f"Expected a directory: {directory}")
    for parent, dirs, names in os.walk(directory, followlinks=False):
        base = Path(parent)
        for name in list(dirs):
            child = base / name
            if base == directory and name in skip_links and child.is_symlink():
                dirs.remove(name)
                continue
            _reject_link(child)
            if name in {"__pycache__", ".pytest_cache", ".git", ".ruff_cache"}:
                dirs.remove(name)
        for name in names:
            child = base / name
            if base == directory and name in skip_links and child.is_symlink():
                continue
            _reject_link(child)
            if name.endswith((".pyc", ".pyo")):
                continue
            if not child.is_file():
                raise PortableError(f"Expected a regular file: {child}")
            relative = child.relative_to(directory).as_posix()
            files[_relative(f"{prefix}/{relative}")] = child
    return files


def _source_files(project: Path, model: Path) -> tuple[dict[str, Path], str]:
    project = project.resolve(strict=True)
    model = model.resolve(strict=True)
    sys.path.insert(0, str(project / "src"))
    from cv_human_video_matting.utils.model_bundle import verify_model_bundle

    identity = verify_model_bundle(model)
    files = {}
    for name in SOURCE_FILES:
        path = project / name
        if not path.is_file():
            raise PortableError(f"Required project file missing: {path}")
        _reject_link(path)
        files[name] = path
    for name in SOURCE_DIRS:
        files.update(_walk(project / name, name))
    for relative, path in _walk(model, MODEL_DEST).items():
        files[relative] = path
    if f"{MODEL_DEST}/manifest.json" not in files:
        raise PortableError("Model bundle manifest is missing")
    return files, identity["content_sha256"]


def _backup_files(root: Path) -> dict[str, Path]:
    files = {}
    for name in BACKUP_DIRS:
        # current-rvm is a locally recreated pointer. Preserve its target files,
        # not the symlink's machine-specific absolute path.
        skipped = frozenset({"current-rvm"}) if name == "weights" else frozenset()
        files.update(_walk(root / name, name, skip_links=skipped))
    return files


def _add_file(archive: ZipFile, path: Path, name: str) -> dict:
    _reject_link(path)
    before = path.stat()
    info = ZipInfo(_relative(name), date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as dest:
        for block in iter(lambda: source.read(CHUNK), b""):
            dest.write(block)
            digest.update(block)
            size += len(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
        raise PortableError(f"File changed during packaging: {path}")
    return {"path": name, "sha256": digest.hexdigest(), "size_bytes": size}


def _sidecar(archive: Path) -> Path:
    return archive.with_name(archive.name + ".sha256")


def create_archive(output: Path, files: dict[str, Path], kind: str, **metadata) -> dict:
    output = output.absolute()
    sidecar = _sidecar(output)
    if output.exists() or sidecar.exists():
        raise PortableError(f"Archive or SHA256 sidecar already exists: {output}")
    if kind not in {"deployment", "data_backup"}:
        raise PortableError(f"Unsupported archive kind: {kind}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not files and kind == "deployment":
        raise PortableError("Refusing an empty archive")
    folded = [name.casefold() for name in files]
    if len(folded) != len(set(folded)) or "manifest.json" in folded:
        raise PortableError("Duplicate/reserved archive path")
    descriptor, staged_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    stage = Path(staged_name)
    published = False
    sidecar_created = False
    try:
        records = []
        with ZipFile(stage, "w", allowZip64=True) as archive:
            for name, path in sorted(files.items()):
                records.append(_add_file(archive, path, _relative(name)))
            manifest = {
                "schema_version": 1, "kind": kind,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": records, **metadata,
            }
            info = ZipInfo("manifest.json", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        digest = _hash_path(stage)["sha256"]
        # A hard link publishes without replacing an archive created by another process.
        os.link(stage, output)
        published = True
        stage.unlink()
        with sidecar.open("x", encoding="ascii") as handle:
            sidecar_created = True
            handle.write(f"{digest}  {output.name}\n")
        verify_archive(output, kind)
        return manifest
    except BaseException:
        stage.unlink(missing_ok=True)
        if published:
            output.unlink(missing_ok=True)
        if sidecar_created:
            sidecar.unlink(missing_ok=True)
        raise


def verify_archive(archive_path: Path, expected_kind: str | None = None) -> dict:
    archive_path = archive_path.resolve(strict=True)
    sidecar = _sidecar(archive_path)
    try:
        contents = sidecar.read_text(encoding="ascii").strip()
        digest, filename = contents.split("  ", 1)
    except (OSError, UnicodeError, ValueError) as exc:
        raise PortableError(f"Missing/invalid SHA256 sidecar: {sidecar}") from exc
    if not HASH_PATTERN.fullmatch(digest) or filename != archive_path.name:
        raise PortableError(f"Invalid SHA256 sidecar: {sidecar}")
    if _hash_path(archive_path)["sha256"] != digest:
        raise PortableError("Archive SHA256 does not match its sidecar")
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = []
        for info in infos:
            name = _relative(info.filename)
            mode = info.external_attr >> 16
            if info.is_dir() or (mode and not stat.S_ISREG(mode)):
                raise PortableError(f"Non-file ZIP entry is forbidden: {name}")
            names.append(name)
        if len(names) != len(set(name.casefold() for name in names)):
            raise PortableError("Duplicate archive paths")
        if names.count("manifest.json") != 1:
            raise PortableError("Missing archive manifest")
        info = archive.getinfo("manifest.json")
        if info.file_size > 4 * CHUNK:
            raise PortableError("Manifest is too large")
        manifest = json.loads(archive.read(info))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise PortableError("Unsupported archive manifest")
        kind = manifest.get("kind")
        if kind not in {"deployment", "data_backup"} or (expected_kind and kind != expected_kind):
            raise PortableError(f"Wrong archive kind: {kind}")
        entries = manifest.get("files")
        if not isinstance(entries, list) or (kind == "deployment" and not entries):
            raise PortableError("Archive has no file inventory")
        expected = {}
        for item in entries:
            if not isinstance(item, dict):
                raise PortableError("Invalid file inventory")
            name = _relative(item.get("path"))
            if (name == "manifest.json" or name in expected or not isinstance(item.get("sha256"), str)
                    or not HASH_PATTERN.fullmatch(item["sha256"])
                    or type(item.get("size_bytes")) is not int or item["size_bytes"] < 0):
                raise PortableError(f"Invalid file inventory record: {name}")
            if kind == "data_backup" and not any(name.startswith(prefix + "/") for prefix in BACKUP_DIRS):
                raise PortableError(f"Backup contains unexpected path: {name}")
            expected[name] = item
        if set(names) != set(expected) | {"manifest.json"}:
            raise PortableError("Archive entries differ from its manifest")
        for name, item in expected.items():
            with archive.open(name) as source:
                actual = _file_info(source)
            if actual != {"sha256": item["sha256"], "size_bytes": item["size_bytes"]}:
                raise PortableError(f"Archive file changed: {name}")
        return manifest


def extract_archive(archive_path: Path, target: Path, expected_kind: str = "deployment") -> dict:
    manifest = verify_archive(archive_path, expected_kind)
    target = target.absolute()
    if target.exists():
        raise PortableError(f"Extraction target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        with ZipFile(archive_path) as archive:
            for item in manifest["files"]:
                name = _relative(item["path"])
                destination = stage.joinpath(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, CHUNK)
        if target.exists():
            raise PortableError(f"Extraction target appeared: {target}")
        stage.rename(target)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def restore_backup(archive_path: Path, root: Path) -> dict:
    manifest = verify_archive(archive_path, "data_backup")
    _reject_link(root)
    root = root.resolve(strict=True)
    pending = []
    for item in manifest["files"]:
        destination = root.joinpath(*PurePosixPath(item["path"]).parts)
        for parent in (destination, *destination.parents):
            if parent == root.parent:
                break
            if parent.exists() or parent.is_symlink():
                _reject_link(parent)
        if destination.exists():
            if not destination.is_file() or _hash_path(destination) != {
                "sha256": item["sha256"], "size_bytes": item["size_bytes"]
            }:
                raise PortableError(f"Restore would overwrite different data: {destination}")
        else:
            pending.append((item["path"], destination))
    with ZipFile(archive_path) as archive:
        for name, destination in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, staged_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            staged = Path(staged_name)
            try:
                with archive.open(name) as source, os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(source, output, CHUNK)
                os.link(staged, destination)
            finally:
                staged.unlink(missing_ok=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    pack.add_argument("--model", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--target", type=Path, required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--root", type=Path, required=True)
    backup.add_argument("--output", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            files, identity = _source_files(args.project, args.model)
            result = create_archive(args.output, files, "deployment", model_content_sha256=identity,
                                    model_relative_path=MODEL_DEST)
        elif args.command == "verify":
            result = verify_archive(args.archive)
        elif args.command == "extract":
            result = extract_archive(args.archive, args.target)
        elif args.command == "backup":
            if args.output.absolute().resolve().is_relative_to(args.root.resolve(strict=True)):
                raise PortableError("Save backups outside shared/data and shared/logs")
            result = create_archive(args.output, _backup_files(args.root), "data_backup")
        elif args.command == "restore":
            result = restore_backup(args.archive, args.root)
    except (PortableError, OSError, UnicodeError, ValueError, KeyError, BadZipFile) as error:
        parser.exit(1, f"Portable deployment failed: {error}\n")
    print(json.dumps({"kind": result["kind"], "files": len(result["files"]),
                      "model_content_sha256": result.get("model_content_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
