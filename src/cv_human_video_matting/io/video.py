"""Video probing and simple frame-copy utilities."""

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


def _checked_video(path: Path) -> VideoInfo:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Video is missing or empty: {path}")
    info = probe_video(path)
    if (
        info.width <= 0
        or info.height <= 0
        or info.frame_count <= 0
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


def export_video_for_delivery(
    processed_path: str | Path,
    output_path: str | Path,
    source_path: str | Path | None = None,
    preserve_audio: bool = True,
) -> Path:
    """Atomically export H.264/AAC MP4, keeping audio within the processed duration.

    ``source_path`` supplies the original video's optional first audio track. If
    omitted, audio is taken from ``processed_path``. A silent source is valid.
    Odd video dimensions are padded by at most one pixel for yuv420p support.
    The processed file may be replaced in place, but an explicitly supplied
    original source can never be used as the output. Conversion failures raise
    an error and leave any existing output untouched.
    """
    processed = Path(processed_path)
    output = Path(output_path)
    original = Path(source_path) if source_path is not None else None
    if original is not None and _same_file(original, output):
        raise ValueError("Delivery output must not overwrite the original source video.")
    info = _checked_video(processed)
    if preserve_audio and original is not None:
        _checked_video(original)

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
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Could not run FFmpeg for video export: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg video export failed: {detail[-2000:]}")
        try:
            _checked_video(temporary)
        except (OSError, ValueError) as exc:
            raise RuntimeError("FFmpeg did not produce a valid, nonempty video.") from exc
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
    """Return a browser-compatible copy, preserving audio when available.

    Legacy callers fall back to the input when conversion is unavailable. Use
    ``strict=True`` to report conversion failures instead. Cached and freshly
    converted copies must contain valid video frames before being returned.
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
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Could not create a browser-compatible video: {exc}") from exc
        return source
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
