"""Build and verify a local defense candidate from committed Git sources.

Uses only the Python standard library and local Git; never downloads models.
The destination is published only after the archive and history bundle verify.
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
import subprocess
import sys
import tempfile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo


CHUNK_SIZE = 1024 * 1024
GIT_TIMEOUT_SECONDS = 300


class PackagingError(ValueError):
    """The repository or requested evidence cannot be packaged safely."""


def _git(repo: Path, *args: str) -> bytes:
    repo = repo.resolve()
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
            capture_output=True, check=False, timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise PackagingError(
            f"Git {args[0]} timed out after {GIT_TIMEOUT_SECONDS} seconds; no package was published."
        ) from error
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise PackagingError(f"Git {args[0]} failed: {detail}")
    return result.stdout


def _hash_stream(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _file_info(path: Path) -> dict:
    with path.open("rb") as stream:
        digest, size = _hash_stream(stream)
    return {"sha256": digest, "size_bytes": size}


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_links(path: Path) -> None:
    # Inspect before resolve(): resolving a link first hides its presence.
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400,
        )
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise PackagingError(f"Symbolic links/junctions are not supported: {item}")


def _archive_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise PackagingError(f"Unsafe archive path: {name!r}")
    return name


def _assert_clean(repo: Path) -> None:
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=no"):
        raise PackagingError("Tracked files have uncommitted changes; commit them before packaging.")


def _repository_state(repo: Path) -> tuple[str, str, bytes]:
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    refs = _git(repo, "show-ref", "--head")
    return commit, branch, refs


def _project_version(repo: Path, commit: str) -> str:
    contents = _git(repo, "show", f"{commit}:pyproject.toml").decode("utf-8-sig")
    section = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", contents)
    match = re.search(r"(?m)^version\s*=\s*['\"]([^'\"]+)['\"]", section[1] if section else "")
    if not match:
        raise PackagingError("Committed pyproject.toml must contain a static [project].version.")
    return match[1]


def _evidence_files(paths: list[Path], output: Path) -> list[tuple[Path, str]]:
    def walk_error(error):
        raise PackagingError(f"Cannot enumerate evidence: {error}") from error

    files = []
    roots = set()
    names = set()
    for supplied in paths:
        path = supplied.expanduser().absolute()
        _reject_links(path)
        path = path.resolve(strict=True)
        if _within(output, path) or _within(path, output):
            raise PackagingError("Evidence and output paths must not contain one another.")
        root = _archive_name(f"evidence/{path.name}")
        if root.casefold() in roots:
            raise PackagingError(f"Evidence names collide: {path.name}")
        roots.add(root.casefold())
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = []
            for directory, subdirs, filenames in os.walk(path, followlinks=False, onerror=walk_error):
                for name in sorted(subdirs + filenames):
                    child = Path(directory) / name
                    _reject_links(child)
                    if not _within(child.resolve(), path):
                        raise PackagingError(f"Evidence escapes its root: {child}")
                candidates.extend(Path(directory) / name for name in sorted(filenames))
        else:
            raise PackagingError(f"Evidence is not a regular file or directory: {path}")
        for file in sorted(candidates):
            if not stat.S_ISREG(file.stat().st_mode):
                raise PackagingError(f"Evidence is not a regular file: {file}")
            relative = file.relative_to(path).as_posix() if path.is_dir() else ""
            name = _archive_name(f"{root}/{relative}" if relative else root)
            key = name.casefold()
            if key in names:
                raise PackagingError(f"Evidence archive paths collide: {name}")
            names.add(key)
            files.append((file, name))
    # A file supplied as evidence/x must not also be a parent of evidence/x/y.
    for key in names:
        if any(str(parent).casefold() in names for parent in PurePosixPath(key).parents):
            raise PackagingError(f"Evidence file/directory collision: {key}")
    return sorted(files, key=lambda item: item[1])


def _verify_zip(path: Path, expected: dict | None = None) -> dict:
    entries = {}
    all_names = set()
    with ZipFile(path) as archive:
        for info in archive.infolist():
            name = _archive_name(info.filename.rstrip("/"))
            key = name.casefold()
            if key in all_names:
                raise PackagingError(f"Duplicate ZIP entry: {name}")
            all_names.add(key)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise PackagingError(f"Source ZIP contains a symbolic link: {name}")
            if info.is_dir():
                continue
            with archive.open(info) as stream:
                digest, size = _hash_stream(stream)  # ZipExtFile also verifies CRC.
            if size != info.file_size:
                raise PackagingError(f"Incorrect ZIP entry size: {name}")
            entries[name] = {"sha256": digest, "size_bytes": size}
        failed = archive.testzip()
        if failed is not None:
            raise PackagingError(f"ZIP CRC verification failed: {failed}")
    if expected is not None and entries != expected:
        raise PackagingError("ZIP content SHA256 verification failed.")
    return entries


def _append_file(archive: ZipFile, path: Path, name: str, date_time: tuple) -> None:
    _reject_links(path)
    info = ZipInfo(_archive_name(name), date_time=date_time)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        shutil.copyfileobj(source, target, CHUNK_SIZE)


def _start_here(version: str, commit: str, branch: str, evidence_count: int) -> str:
    return f"""# 本地答辩候选包

版本：{version}
源码提交：{commit}
分支：{branch}

这是从已提交代码生成的本地候选交付物，不等于已在远端发布 Release。
包内没有 Python 运行环境或已安装依赖。source/ 不含模型权重；evidence/ 仅包含
打包时明确选入的证据，可能包含单独准备的离线模型包，以 manifest.json 为准。
运行前请阅读 source/README.md 及 evidence/ 中的说明，按要求准备依赖与模型。
离线测试通过不能替代真实人像质量与摄像头验证。

## 内容与使用

- candidate.zip：解压后 source/ 为指定提交的源码；evidence/ 为明确选入的 {evidence_count} 个证据文件。
- repository.bundle：本地 Git 的完整可达引用历史（含本地其他分支）；不会访问或同步远端。
- START_HERE.md：本说明。
- manifest.json：包目录外置清单，记录提交、版本、文件大小、SHA256 和 ZIP 内逐文件 SHA256。
  清单自身不参与哈希，避免循环引用；不要将清单当作数字签名。

在解压目录执行 `git clone repository.bundle restored-project` 可恢复仓库历史。
如需精确回到答辩源码，进入 restored-project 后执行 `git checkout {commit}`。
直接运行可进入 source/，按 README 的环境准备、测试和启动步骤操作。

交付前可使用文件 SHA256 与 manifest.json 对照；全部证据须按各自报告的日期、
模型、输入素材与计时口径解读。未选入的文件和未跟踪工作文件不属于此源码交付。
"""


def package_release(repo: Path, output_dir: Path, evidence: list[Path] | None = None) -> Path:
    repo = repo.resolve(strict=True)
    actual_root = Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if repo != actual_root:
        raise PackagingError("The supplied repository must be its Git working-tree root.")
    output = output_dir.expanduser().absolute()
    _reject_links(output)
    output = output.resolve()
    if output.exists() or output == output.parent:
        raise PackagingError(f"Output already exists or is a filesystem root: {output}")
    if _within(output, repo / ".git"):
        raise PackagingError("Output must not be inside Git metadata.")
    _assert_clean(repo)
    state = _repository_state(repo)
    commit, branch, _ = state
    if _git(repo, "rev-parse", "--is-shallow-repository").strip() == b"true":
        raise PackagingError("A shallow repository cannot provide the complete history bundle.")
    tree = _git(repo, "ls-tree", "-r", "-z", commit)
    if any(entry.startswith(b"160000 ") for entry in tree.split(b"\0")):
        raise PackagingError("Submodule contents require separate packaging; refusing incomplete sources.")
    version = _project_version(repo, commit)
    untracked = len([name for name in _git(
        repo, "ls-files", "--others", "--exclude-standard", "-z",
    ).split(b"\0") if name])
    if untracked:
        print(f"Reminder: {untracked} untracked file(s) excluded from source/.", file=sys.stderr)
    selected = _evidence_files(evidence or [], output)
    evidence_info = {name: _file_info(path) for path, name in selected}
    # Stable ZIP timestamps derive from the source commit, not the packaging clock.
    timestamp = int(_git(repo, "show", "-s", "--format=%ct", commit))
    date = datetime.fromtimestamp(timestamp, timezone.utc)
    date_time = (max(1980, min(date.year, 2107)), date.month, date.day,
                 date.hour, date.minute, date.second)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        archive_path = staging / "candidate.zip"
        bundle_path = staging / "repository.bundle"
        guide_path = staging / "START_HERE.md"
        _git(repo, "archive", "--format=zip", "--prefix=source/", f"--output={archive_path}", commit)
        expected = _verify_zip(archive_path)
        if any(not name.startswith("source/") for name in expected):
            raise PackagingError("Git archive contains a path outside source/.")
        _git(repo, "bundle", "create", str(bundle_path), "--all")
        _git(repo, "bundle", "verify", str(bundle_path))
        guide_path.write_text(_start_here(version, commit, branch, len(selected)), encoding="utf-8")
        files = {path.name: _file_info(path) for path in (bundle_path, guide_path)}
        with ZipFile(archive_path, "a", compression=ZIP_DEFLATED) as archive:
            for path, name in selected:
                _append_file(archive, path, name, date_time)
            for path in (bundle_path, guide_path):
                _append_file(archive, path, path.name, date_time)
        expected.update(evidence_info)
        expected.update(files)
        entries = _verify_zip(archive_path, expected)
        files[archive_path.name] = _file_info(archive_path)
        manifest = {
            "schema_version": 1,
            "package_kind": "local_defense_candidate",
            "remote_release": False,
            "commit": commit,
            "branch": branch,
            "package_version": version,
            "untracked_files_excluded": untracked,
            "files": files,
            "evidence": evidence_info,
            "zip_entries": entries,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _assert_clean(repo)
        if _repository_state(repo) != state:
            raise PackagingError("Git references changed during packaging; retry from a stable repository.")
        if output.exists():
            raise PackagingError(f"Output appeared during packaging; refusing to replace it: {output}")
        staging.rename(output)
        return output
    finally:
        # Only this invocation's sibling staging tree can be removed.
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New package directory; an existing target is never overwritten.")
    parser.add_argument("--evidence", action="append", default=[], type=Path,
                        help="Explicit evidence file/directory; may be repeated.")
    args = parser.parse_args(argv)
    try:
        output = package_release(Path(__file__).resolve().parents[1], args.output_dir, args.evidence)
    except (PackagingError, OSError, BadZipFile) as error:
        parser.exit(1, f"Package failed: {error}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
