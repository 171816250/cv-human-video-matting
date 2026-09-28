"""Verify odd-sized inputs survive video encoding without lost edge pixels."""

import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from cv_human_video_matting import pipeline
from cv_human_video_matting.io.video import probe_video
from cv_human_video_matting.matting.base import MattingResult


class IdentityRunner:
    def predict(self, frame):
        return MattingResult(np.full(frame.shape[:2], 255, dtype=np.uint8), frame.copy())


def make_odd_video(path):
    frame = np.zeros((25, 33, 3), dtype=np.uint8)
    frame[:] = (10, 80, 20)
    frame[-1, :] = (0, 0, 250)
    frame[:, -1] = (250, 0, 0)
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "33x25",
         "-framerate", "12", "-i", "pipe:0", "-c:v", "ffv1", str(path)],
        input=frame.tobytes() * 2, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return frame


@pytest.mark.parametrize("mode", ["green", "side_by_side"])
@pytest.mark.parametrize("resize_short", [None, 60])
def test_video_padding_preserves_original_edge_pixels(
    tmp_path, monkeypatch, mode, resize_short
):
    source, target = tmp_path / "odd.avi", tmp_path / "output.mp4"
    original = make_odd_video(source)
    encoded_frames = []
    real_writer = cv2.VideoWriter

    class RecordingWriter:
        def __init__(self, *args):
            self.writer = real_writer(*args)

        def isOpened(self):
            return self.writer.isOpened()

        def write(self, frame):
            encoded_frames.append(frame.copy())
            self.writer.write(frame)

        def release(self):
            self.writer.release()

    monkeypatch.setattr(pipeline.cv2, "VideoWriter", RecordingWriter)
    stats = pipeline.process_video(
        source, target, IdentityRunner(), mode, resize_short=resize_short,
    )

    expected = np.hstack((original, original)) if mode == "side_by_side" else original
    target_width = 66 if mode == "side_by_side" else 34
    assert stats.frames == len(encoded_frames) == 2
    for rendered in encoded_frames:
        assert rendered.shape == (26, target_width, 3)
        np.testing.assert_array_equal(rendered[:25, :expected.shape[1]], expected)
        assert not rendered[25].any()
        if mode == "green":
            assert not rendered[:, 33].any()
    info = probe_video(target)
    assert (info.width, info.height, info.frame_count) == (target_width, 26, 2)


@pytest.mark.parametrize("resize_short", [None, 60])
def test_transparent_frames_keep_exact_odd_dimensions_and_edges(tmp_path, resize_short):
    source = tmp_path / "odd.avi"
    original = make_odd_video(source)

    stats = pipeline.process_video(
        source, tmp_path / "transparent.mp4", IdentityRunner(), "transparent",
        resize_short=resize_short,
    )

    assert stats.frames == 2
    for path in stats.output_path.glob("frame_*.png"):
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert frame.shape == (25, 33, 4)
        np.testing.assert_array_equal(frame[:, :, :3], original)
        assert np.all(frame[:, :, 3] == 255)
