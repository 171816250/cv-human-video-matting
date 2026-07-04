"""Tests for extended compositor functions (blur and transparent modes)."""

import numpy as np
import pytest

from cv_human_video_matting.render.compositor import (
    compose_on_blur,
    compose_transparent,
)


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_alpha():
    """Create a sample alpha matte."""
    alpha = np.zeros((480, 640), dtype=np.float32)
    alpha[140:340, 220:420] = 0.9  # Person region
    return alpha


class TestComposeOnBlur:
    """Tests for background blur composition."""

    def test_compose_on_blur_basic(self, sample_frame, sample_alpha):
        """Test basic blur composition."""
        result = compose_on_blur(sample_frame, sample_alpha, blur_ksize=45)

        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_compose_on_blur_even_kernel(self, sample_frame, sample_alpha):
        """Test that even kernel size is converted to odd."""
        # Should not raise error even with even kernel size
        result = compose_on_blur(sample_frame, sample_alpha, blur_ksize=44)

        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

    def test_compose_on_blur_small_kernel(self, sample_frame, sample_alpha):
        """Test with small blur kernel."""
        result = compose_on_blur(sample_frame, sample_alpha, blur_ksize=5)

        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

    def test_compose_on_blur_large_kernel(self, sample_frame, sample_alpha):
        """Test with large blur kernel."""
        result = compose_on_blur(sample_frame, sample_alpha, blur_ksize=99)

        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

    def test_compose_on_blur_zero_alpha(self, sample_frame):
        """Test with zero alpha (all background)."""
        alpha = np.zeros((480, 640), dtype=np.float32)
        result = compose_on_blur(sample_frame, alpha, blur_ksize=45)

        assert result.shape == sample_frame.shape
        # Result should be fully blurred

    def test_compose_on_blur_full_alpha(self, sample_frame):
        """Test with full alpha (all foreground)."""
        alpha = np.ones((480, 640), dtype=np.float32)
        result = compose_on_blur(sample_frame, alpha, blur_ksize=45)

        assert result.shape == sample_frame.shape
        # Result should be close to original (no blur visible)
        # Allow small differences due to floating point operations
        diff = np.abs(result.astype(float) - sample_frame.astype(float))
        assert np.mean(diff) < 2.0

    def test_compose_on_blur_3d_alpha(self, sample_frame, sample_alpha):
        """Test with 3D alpha (HxWx1)."""
        alpha_3d = sample_alpha[:, :, None]
        result = compose_on_blur(sample_frame, alpha_3d, blur_ksize=45)

        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8


class TestComposeTransparent:
    """Tests for transparent BGRA composition."""

    def test_compose_transparent_basic(self, sample_frame, sample_alpha):
        """Test basic transparent composition."""
        result = compose_transparent(sample_frame, sample_alpha)

        assert result.shape == (sample_frame.shape[0], sample_frame.shape[1], 4)
        assert result.dtype == np.uint8
        # Check that RGB channels match input
        assert np.array_equal(result[:, :, :3], sample_frame)

    def test_compose_transparent_alpha_channel(self, sample_frame, sample_alpha):
        """Test that alpha channel is correctly set."""
        result = compose_transparent(sample_frame, sample_alpha)

        alpha_channel = result[:, :, 3]
        assert alpha_channel.min() >= 0
        assert alpha_channel.max() <= 255

        # Check person region has high alpha
        person_alpha = alpha_channel[140:340, 220:420]
        assert np.mean(person_alpha) > 200  # Should be close to 255 (0.9 * 255)

    def test_compose_transparent_zero_alpha(self, sample_frame):
        """Test with zero alpha (fully transparent)."""
        alpha = np.zeros((480, 640), dtype=np.float32)
        result = compose_transparent(sample_frame, alpha)

        alpha_channel = result[:, :, 3]
        assert np.all(alpha_channel == 0)

    def test_compose_transparent_full_alpha(self, sample_frame):
        """Test with full alpha (fully opaque)."""
        alpha = np.ones((480, 640), dtype=np.float32)
        result = compose_transparent(sample_frame, alpha)

        alpha_channel = result[:, :, 3]
        assert np.all(alpha_channel == 255)

    def test_compose_transparent_3d_alpha(self, sample_frame, sample_alpha):
        """Test with 3D alpha (HxWx1)."""
        alpha_3d = sample_alpha[:, :, None]
        result = compose_transparent(sample_frame, alpha_3d)

        assert result.shape == (sample_frame.shape[0], sample_frame.shape[1], 4)
        assert result.dtype == np.uint8

    def test_compose_transparent_bgr_preserved(self, sample_frame, sample_alpha):
        """Test that BGR channels are preserved."""
        result = compose_transparent(sample_frame, sample_alpha)

        # BGR channels should be identical to input
        assert np.array_equal(result[:, :, :3], sample_frame)

    def test_compose_transparent_alpha_range(self, sample_frame):
        """Test with various alpha values."""
        # Gradient alpha from 0 to 1
        alpha = np.linspace(0, 1, 640, dtype=np.float32)
        alpha = np.tile(alpha, (480, 1))

        result = compose_transparent(sample_frame, alpha)

        alpha_channel = result[:, :, 3]
        # Should have gradient from 0 to 255
        assert alpha_channel[0, 0] < 5
        assert alpha_channel[0, -1] > 250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
