from pathlib import Path

import cv2
import numpy as np

from cv_human_video_matting.io.video import ensure_browser_playable_video, probe_video


def _write_mp4v_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (32, 24))
    assert writer.isOpened()
    try:
        for index in range(3):
            frame = np.full((24, 32, 3), 40 + index * 40, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_ensure_browser_playable_video_returns_original_when_converter_missing(tmp_path):
    input_path = tmp_path / "input.mp4"
    _write_mp4v_video(input_path)

    output = ensure_browser_playable_video(input_path, converter=None)

    assert output == input_path


def test_ensure_browser_playable_video_uses_converter_output(tmp_path):
    input_path = tmp_path / "input.mp4"
    _write_mp4v_video(input_path)

    def fake_converter(source: Path, target: Path) -> bool:
        _write_mp4v_video(target)
        return True

    output = ensure_browser_playable_video(input_path, converter=fake_converter)

    assert output.name == "input_browser.mp4"
    assert output.exists()
    assert probe_video(output).frame_count == 3
