"""Create a tiny synthetic video for the video I/O smoke test."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    output = Path("data/input/smoke_demo.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    background_output = Path("data/backgrounds/smoke_bg.jpg")
    background_output.parent.mkdir(parents=True, exist_ok=True)

    width, height, fps, frame_count = 320, 180, 24, 48
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create smoke video: {output}")

    try:
        for frame_index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = np.linspace(30, 160, width, dtype=np.uint8)
            frame[:, :, 1] = np.linspace(40, 120, height, dtype=np.uint8)[:, None]

            center_x = int(width * 0.5 + math.sin(frame_index / 8) * 70)
            center_y = int(height * 0.52)
            skin = (225, 205, 185)
            shirt = (60, 180, 230)

            cv2.circle(frame, (center_x, center_y - 38), 20, skin, -1)
            cv2.rectangle(frame, (center_x - 24, center_y - 15), (center_x + 24, center_y + 50), shirt, -1)
            cv2.line(frame, (center_x - 24, center_y + 8), (center_x - 60, center_y + 34), skin, 6)
            cv2.line(frame, (center_x + 24, center_y + 8), (center_x + 60, center_y + 34), skin, 6)
            cv2.putText(
                frame,
                f"frame {frame_index:02d}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    background = np.zeros((240, 426, 3), dtype=np.uint8)
    background[:, :, 0] = np.linspace(50, 180, background.shape[1], dtype=np.uint8)
    background[:, :, 1] = np.linspace(30, 120, background.shape[0], dtype=np.uint8)[:, None]
    background[:, :, 2] = 90
    cv2.circle(background, (330, 60), 34, (40, 220, 240), -1)
    cv2.rectangle(background, (0, 170), (426, 240), (70, 95, 45), -1)
    cv2.line(background, (0, 170), (426, 130), (120, 150, 90), 3)
    if not cv2.imwrite(str(background_output), background):
        raise RuntimeError(f"Failed to create smoke background: {background_output}")

    print(output)
    print(background_output)


if __name__ == "__main__":
    main()
