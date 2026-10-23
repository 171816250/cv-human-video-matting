"""发布已准备好的结果文件，普通异常时尝试回滚。

替换前先备份旧文件；单个文件的替换是原子的。
多个文件不构成一次文件系统事务，不保证断电或崩溃时整体回滚。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import tempfile


@dataclass(frozen=True)
class PublicationItem:
    staged: Path
    destination: Path
    replace_existing: bool = True


def _identity(path: Path):
    if not path.exists():
        return None
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _validate_items(items: list[PublicationItem]) -> None:
    destinations = set()
    for item in items:
        source, target = Path(item.staged), Path(item.destination)
        if source.is_symlink() or not (source.is_file() or source.is_dir()):
            raise ValueError(f"Staged artifact is missing or not a regular local artifact: {source}")
        resolved = target.resolve()
        if resolved in destinations:
            raise ValueError(f"Duplicate artifact destination: {target}")
        destinations.add(resolved)
        if resolved == source.resolve() or resolved.is_relative_to(source.resolve()):
            raise ValueError(f"Artifact destination must be outside its staged source: {target}")
        if target.is_symlink():
            raise ValueError(f"Artifact destination cannot be a symbolic link: {target}")
        if target.exists():
            if source.is_dir() or not item.replace_existing:
                raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
            if not target.is_file():
                raise ValueError(f"Artifact destination is not a file: {target}")
        for parent in target.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Artifact parent is not a directory: {parent}")
    for target in destinations:
        if any(parent in destinations for parent in target.parents):
            raise ValueError(f"Artifact destinations must not contain one another: {target}")


def _make_parents(parent: Path, created: list[Path]) -> None:
    missing = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
            created.append(directory)
        except FileExistsError:
            if not directory.is_dir():
                raise


def publish_artifacts(items: list[PublicationItem]) -> None:
    """先准备副本和备份，再逐项发布；失败时恢复旧文件。

    新文件以独占方式创建，已有文件原子替换。
    仅回滚本次发布且未被其他程序修改的文件；无法恢复时保留备份位置。
    """
    items = [PublicationItem(Path(item.staged), Path(item.destination), item.replace_existing)
             for item in items]
    _validate_items(items)
    preparations = []
    published = []
    created_parents = []
    recovery = set()
    success = False
    try:
        for item in items:
            target = item.destination
            _make_parents(target.parent, created_parents)
            temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=target.parent)).resolve()
            record = {"item": item, "temporary": temporary, "original": _identity(target),
                      "new": temporary / "new", "backup": temporary / "previous"}
            preparations.append(record)
            if record["original"] is not None and (item.staged.is_dir() or not item.replace_existing):
                raise FileExistsError(f"Artifact appeared before preparation: {target}")
            if item.staged.is_dir():
                shutil.copytree(item.staged, record["new"])
            else:
                shutil.copy2(item.staged, record["new"])
            if record["original"] is not None:
                shutil.copy2(target, record["backup"])
                if _identity(target) != record["original"]:
                    raise RuntimeError(f"Destination changed while preparing its backup: {target}")

        # All new artifacts and existing-file backups are ready before changing
        # any user-visible destination.
        for record in preparations:
            target = record["item"].destination
            if _identity(target) != record["original"]:
                raise RuntimeError(f"Destination changed before publication: {target}")
            fresh = record["new"]
            record["published_identity"] = _identity(fresh)
            if record["original"] is not None:
                fresh.replace(target)
            elif fresh.is_dir():
                fresh.rename(target)
            else:
                os.link(fresh, target)  # Exclusive; both paths share a filesystem.
            published.append(record)
            if fresh.exists():
                fresh.unlink()
        success = True
    except BaseException as error:
        problems = []
        for record in reversed(published):
            target = record["item"].destination
            try:
                if _identity(target) != record["published_identity"]:
                    raise RuntimeError(f"Destination changed after publication: {target}")
                if record["original"] is not None:
                    record["backup"].replace(target)
                elif target.is_dir():
                    # Move our exact newly published directory back into our
                    # private tree before cleanup; never recursively delete a
                    # computed user destination.
                    target.rename(record["new"])
                else:
                    target.unlink()
            except BaseException as rollback_error:
                recovery.add(record["temporary"])
                problems.append(f"{target}: {rollback_error}; recovery: {record['temporary']}")
        if problems:
            raise RuntimeError("Publication failed and rollback was incomplete. " + "; ".join(problems)) from error
        raise
    finally:
        for record in preparations:
            temporary = record["temporary"]
            parent = record["item"].destination.parent.resolve()
            if temporary not in recovery and temporary.parent == parent and temporary.name.startswith(".publish-"):
                try:
                    shutil.rmtree(temporary)
                except OSError as cleanup_error:
                    # 临时目录清理失败只记日志，不把成功发布误报为失败，也不覆盖原异常。
                    logging.getLogger(__name__).warning(
                        "Could not clean publication temporary directory %s: %s",
                        temporary, cleanup_error,
                    )
        if not success:
            for directory in reversed(created_parents):
                try:
                    directory.rmdir()
                except OSError:
                    pass
