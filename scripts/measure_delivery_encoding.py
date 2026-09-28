"""Measure intermediate encoding loss on local synthetic frames, without models.

Run from the project environment:
    python scripts/measure_delivery_encoding.py --output-dir data/output/v0.2.0rc2/encoding

The comparison starts from identical decoded BGR frames, as produced by the
matting pipeline. FFV1 stores that reference without pixel loss. H.264 delivery
still uses chroma subsampling and lossy compression. This is an encoding check,
not a measurement of human matting accuracy or perceptual quality.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv_human_video_matting.io.video import export_video_for_delivery


def decode(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video has no decoded frames: {path}")
    return np.stack(frames)


def encode(path: Path, frames: np.ndarray, codec: str, fps: int) -> None:
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {codec} video: {path}")
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def measure(path: Path, reference: np.ndarray) -> dict:
    actual = decode(path)
    if actual.shape != reference.shape:
        raise RuntimeError(f"Frame alignment differs: {actual.shape} vs {reference.shape}")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    return {
        "path": path.name,
        "file_bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "decoded_frames": len(actual),
        "mae_bgr_8bit": float(np.mean(np.abs(difference))),
        "mse_bgr_8bit": mse,
        # Null plus pixel_exact avoids nonstandard JSON Infinity for lossless data.
        "psnr_db_bgr_8bit": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
        "pixel_exact": bool(np.array_equal(actual, reference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/output/v0.2.0rc2/encoding"))
    args = parser.parse_args()
    output = args.output_dir
    # Preserve any earlier evidence; reruns need an explicitly different directory.
    output.mkdir(parents=True, exist_ok=False)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source = output / "synthetic_source.mkv"
    generator = "testsrc2=size=320x180:rate=24:duration=1"
    subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", generator, "-c:v", "ffv1", str(source)],
                   check=True, capture_output=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    frames = decode(source)
    reference_path = output / "reference_bgr_ffv1.mkv"
    old_intermediate = output / "intermediate_mp4v.mp4"
    new_intermediate = output / "intermediate_ffv1.mkv"
    encode(reference_path, frames, "FFV1", 24)
    encode(old_intermediate, frames, "mp4v", 24)
    encode(new_intermediate, frames, "FFV1", 24)
    intermediate_metrics = {
        "mp4v": measure(old_intermediate, frames),
        "ffv1": measure(new_intermediate, frames),
    }
    if not measure(reference_path, frames)["pixel_exact"] or not intermediate_metrics["ffv1"]["pixel_exact"]:
        raise RuntimeError("FFV1 did not preserve the pipeline's BGR pixels exactly")
    delivery = {}
    for name, intermediate in (("direct_h264", reference_path),
                               ("mp4v_then_h264", old_intermediate),
                               ("ffv1_then_h264", new_intermediate)):
        target = export_video_for_delivery(intermediate, output / f"{name}.mp4", preserve_audio=False)
        delivery[name] = measure(target, frames)
    version = subprocess.run([ffmpeg, "-version"], check=True, capture_output=True, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.splitlines()[0]
    evidence = {
        "schema_version": 1,
        "kind": "synthetic_delivery_encoding_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reproduce": "python scripts/measure_delivery_encoding.py --output-dir <new-directory>",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                        "numpy": np.__version__, "ffmpeg": version},
        "input": {"generator": generator, "frames": len(frames), "width": 320, "height": 180,
                  "fps": 24, "reference": "Same decoded BGR frames supplied to all intermediate encoders",
                  "decoded_bgr_sha256": sha256(frames.tobytes()).hexdigest()},
        "delivery_settings": {"codec": "libx264", "crf": 18, "preset": "medium", "pixel_format": "yuv420p"},
        "intermediates": intermediate_metrics,
        "deliveries": delivery,
        "limitations": ["Synthetic color and motion pattern only; no model inference or human video used.",
                        "PSNR/MAE measure decoded BGR pixel error, not matting accuracy or perceived quality.",
                        "H.264 delivery remains lossy; FFV1 only removes the intermediate lossy encoding.",
                        "Codec/library versions can change numerical results and encoded file sizes."],
    }
    path = output / "encoding_comparison.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(path)
    for name, result in delivery.items():
        print(f"{name}: MAE={result['mae_bgr_8bit']:.6f}, PSNR={result['psnr_db_bgr_8bit']:.6f} dB")


if __name__ == "__main__":
    main()
