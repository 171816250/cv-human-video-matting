"""Video probing and simple frame-copy utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
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


def _imageio_ffmpeg_converter(source: Path, target: Path) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        return False

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0 and target.exists() and target.stat().st_size > 0


def ensure_browser_playable_video(
    video_path: str | Path,
    converter: VideoConverter | None = _imageio_ffmpeg_converter,
) -> Path:
    """Return a browser-friendly H.264 MP4 copy when conversion support exists."""
    source = Path(video_path)
    if converter is None:
        return source

    target = source.with_name(f"{source.stem}_browser.mp4")
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    if converter(source, target):
        return target
    return source
