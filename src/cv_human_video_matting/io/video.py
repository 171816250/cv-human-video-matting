"""探测视频信息、读写帧，并导出可在浏览器播放的视频。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Callable

import cv2

from cv_human_video_matting.utils.paths import ensure_parent_dir
from cv_human_video_matting.utils.progress import (
    CancelCallback, ProcessingCancelled, check_cancelled,
)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps

    def to_report(self) -> str:
        return "\n".join(
            [
                f"path: {self.path}",
                f"size: {self.width}x{self.height}",
                f"fps: {self.fps:.3f}",
                f"frames: {self.frame_count}",
                f"duration: {self.duration_seconds:.3f}s",
            ]
        )


def probe_video(path: str | Path) -> VideoInfo:
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Failed to open input video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
    finally:
        capture.release()


def copy_video(input_path: str | Path, output_path: str | Path, max_frames: int | None = None) -> Path:
    info = probe_video(input_path)
    output = Path(output_path)
    ensure_parent_dir(output)

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open input video: {info.path}")

    fps = info.fps if info.fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (info.width, info.height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Failed to open output video writer: {output}")

    processed = 0
    started_at = perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        capture.release()
        writer.release()

    elapsed = perf_counter() - started_at
    effective_fps = processed / elapsed if elapsed > 0 else 0.0
    print(
        f"Processed {processed} frame(s) in {elapsed:.3f}s "
        f"({effective_fps:.2f} FPS copy pipeline)."
    )
    return output


VideoConverter = Callable[[Path, Path], bool]


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def _checked_video(path: Path, *, allow_unknown_frames: bool = False) -> VideoInfo:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Video is missing or empty: {path}")
    info = probe_video(path)
    if (
        info.width <= 0
        or info.height <= 0
        or (not allow_unknown_frames and info.frame_count <= 0)
        or not math.isfinite(info.fps)
        or info.fps <= 0
    ):
        raise ValueError(f"Video has no usable frames or frame rate: {path}")
    return info


def _temporary_mp4(target: Path) -> Path:
    ensure_parent_dir(target)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}-", suffix=".mp4", dir=str(target.parent)
    )
    os.close(descriptor)
    return Path(name)


def _reap_encoder(process: subprocess.Popen) -> None:
    """Terminate unfinished encoders and always collect their exit status."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    else:
        process.wait()


def _run_encoder(command: list[str], should_cancel: CancelCallback | None) -> None:
    # 将编码错误写入临时文件，避免管道写满后阻塞取消检查；报错时只读取末尾。
    with tempfile.TemporaryFile(mode="w+b") as error_log:
        process = None
        try:
            check_cancelled(should_cancel)
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=error_log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            while True:
                check_cancelled(should_cancel)
                try:
                    return_code = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            check_cancelled(should_cancel)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Could not run FFmpeg for video export: {exc}") from exc
        finally:
            # This also runs for KeyboardInterrupt and callback exceptions.
            if process is not None:
                _reap_encoder(process)
        if return_code != 0:
            error_log.seek(0, os.SEEK_END)
            error_log.seek(max(0, error_log.tell() - 2000))
            detail = error_log.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg video export failed: {detail}")


def export_video_for_delivery(
    processed_path: str | Path,
    output_path: str | Path,
    source_path: str | Path | None = None,
    preserve_audio: bool = True,
    should_cancel: CancelCallback | None = None,
) -> Path:
    """导出 H.264/AAC MP4，音轨截到已处理视频的时长。

    优先使用 source_path 的首条音轨，未传入时从处理后视频取音；无音轨也可导出。
    奇数宽高最多补一像素以兼容 yuv420p。允许替换处理后文件，不覆盖显式原视频。
    转换失败保留旧结果；取消时先结束并回收 FFmpeg，再抛出 ProcessingCancelled。
    """
    check_cancelled(should_cancel)
    processed = Path(processed_path)
    output = Path(output_path)
    original = Path(source_path) if source_path is not None else None
    if original is not None and _same_file(original, output):
        raise ValueError("Delivery output must not overwrite the original source video.")
    info = _checked_video(processed)
    if preserve_audio and original is not None:
        # Original audio can be demuxed even when its container cannot report a
        # total video frame count. Duration is always bounded by processed output.
        _checked_video(original, allow_unknown_frames=True)

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("H.264 export requires a working imageio-ffmpeg installation.") from exc

    temporary = _temporary_mp4(output)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(processed),
    ]
    audio_input = "0"
    if preserve_audio and original is not None and not _same_file(original, processed):
        command.extend(["-i", str(original)])
        audio_input = "1"
    command.extend(["-map", "0:v:0"])
    if preserve_audio:
        command.extend(["-map", f"{audio_input}:a:0?", "-c:a", "aac", "-b:a", "192k"])
    else:
        command.append("-an")
    command.extend([
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-t",
        f"{info.duration_seconds:.9f}",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(temporary),
    ])
    try:
        _run_encoder(command, should_cancel)
        try:
            delivered_info = _checked_video(temporary)
        except (OSError, ValueError) as exc:
            raise RuntimeError("FFmpeg did not produce a valid, nonempty video.") from exc
        if delivered_info.frame_count != info.frame_count:
            raise RuntimeError(
                f"FFmpeg output frame count mismatch: expected {info.frame_count}, "
                f"found {delivered_info.frame_count}; existing delivery output was preserved."
            )
        check_cancelled(should_cancel)
        temporary.replace(output)
        return output
    finally:
        temporary.unlink(missing_ok=True)


def _imageio_ffmpeg_converter(source: Path, target: Path) -> bool:
    export_video_for_delivery(source, target)
    return True


def ensure_browser_playable_video(
    video_path: str | Path,
    converter: VideoConverter | None = _imageio_ffmpeg_converter,
    *,
    strict: bool = False,
) -> Path:
    """返回可在浏览器播放的视频副本，并尽量保留音轨。

    默认转换失败时返回原文件；strict=True 时直接报错。
    缓存副本和新副本都须通过有效视频帧检查。
    """
    source = Path(video_path)
    if converter is None:
        if strict:
            raise RuntimeError("No browser-compatible video converter is available.")
        return source

    target = source.with_name(f"{source.stem}_browser.mp4")
    temporary = None
    try:
        _checked_video(source)
        if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            try:
                _checked_video(target)
                return target
            except (OSError, ValueError):
                pass
        temporary = _temporary_mp4(target)
        if not converter(source, temporary):
            raise RuntimeError("The video converter reported a failure.")
        _checked_video(temporary)
        temporary.replace(target)
        return target
    except ProcessingCancelled:
        raise
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Could not create a browser-compatible video: {exc}") from exc
        return source
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
