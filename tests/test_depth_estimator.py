"""Tests for depth estimation and distance labeling."""

import numpy as np
import pytest

from cv_human_video_matting.depth import DepthEstimator, draw_distance_label


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_alpha():
    """Create a sample alpha matte with a centered person region."""
    alpha = np.zeros((480, 640), dtype=np.float32)
    # Create a centered rectangular person region
    alpha[140:340, 220:420] = 0.9
    return alpha


class TestDepthEstimator:
    """Tests for DepthEstimator class."""

    def test_estimator_initialization(self):
        """Test that DepthEstimator initializes correctly."""
        estimator = DepthEstimator(device="cpu", model_type="MiDaS_small")
        assert estimator.device == "cpu"
        assert estimator.model_type == "MiDaS_small"
        assert estimator.model is not None
        assert estimator.transform is not None

    def test_estimate_output_shape(self, sample_frame):
        """Test that depth estimation returns correct shape."""
        estimator = DepthEstimator(device="cpu")
        depth = estimator.estimate(sample_frame)

        assert depth.shape == sample_frame.shape[:2]
        assert depth.dtype == np.float32

    def test_estimate_output_range(self, sample_frame):
        """Test that depth values are normalized to 0~1 range."""
        estimator = DepthEstimator(device="cpu")
        depth = estimator.estimate(sample_frame)

        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_person_distance_with_valid_region(self, sample_alpha):
        """Test distance calculation with valid person region."""
        estimator = DepthEstimator(device="cpu")
        # Create mock depth map
        depth = np.random.rand(480, 640).astype(np.float32)

        distance, label = estimator.person_distance(depth, sample_alpha, thresh=0.5)

        assert distance is not None
        assert isinstance(distance, float)
        assert 0.0 <= distance <= 1.0
        assert label in ["Near", "Mid", "Far"]

    def test_person_distance_near_classification(self, sample_alpha):
        """Test that high inverse depth is classified as Near."""
        estimator = DepthEstimator(device="cpu")
        # Create depth map with high values in person region (close distance)
        depth = np.ones((480, 640), dtype=np.float32) * 0.8

        distance, label = estimator.person_distance(depth, sample_alpha)

        assert label == "Near"
        assert distance < 0.5  # High inverse depth = low distance value

    def test_person_distance_far_classification(self, sample_alpha):
        """Test that low inverse depth is classified as Far."""
        estimator = DepthEstimator(device="cpu")
        # Create depth map with low values in person region (far distance)
        depth = np.ones((480, 640), dtype=np.float32) * 0.2

        distance, label = estimator.person_distance(depth, sample_alpha)

        assert label == "Far"
        assert distance > 0.5  # Low inverse depth = high distance value

    def test_person_distance_no_person(self):
        """Test that no person is detected with empty alpha."""
        estimator = DepthEstimator(device="cpu")
        depth = np.random.rand(480, 640).astype(np.float32)
        alpha = np.zeros((480, 640), dtype=np.float32)  # No person

        distance, label = estimator.person_distance(depth, alpha)

        assert distance is None
        assert label == "No person detected"

    def test_person_distance_threshold(self, sample_alpha):
        """Test alpha threshold filtering."""
        estimator = DepthEstimator(device="cpu")
        depth = np.random.rand(480, 640).astype(np.float32)

        # With high threshold, should not detect person
        distance, label = estimator.person_distance(depth, sample_alpha, thresh=0.95)
        assert distance is None or label == "No person detected"


class TestDistanceVisualization:
    """Tests for distance label visualization."""

    def test_draw_distance_label_basic(self, sample_frame, sample_alpha):
        """Test that distance label is drawn without errors."""
        output = draw_distance_label(
            sample_frame, sample_alpha, distance=0.3, label="Near", thresh=0.5
        )

        assert output.shape == sample_frame.shape
        assert output.dtype == np.uint8
        # Verify output is modified (not identical to input)
        assert not np.array_equal(output, sample_frame)

    def test_draw_distance_label_no_person(self, sample_frame):
        """Test drawing with no person region."""
        alpha = np.zeros((480, 640), dtype=np.float32)
        output = draw_distance_label(sample_frame, alpha, None, "No person")

        # Should return unchanged frame
        assert np.array_equal(output, sample_frame)

    def test_draw_distance_label_with_3d_alpha(self, sample_frame, sample_alpha):
        """Test that 3D alpha (HxWx1) is handled correctly."""
        alpha_3d = sample_alpha[:, :, None]
        output = draw_distance_label(sample_frame, alpha_3d, 0.5, "Mid")

        assert output.shape == sample_frame.shape
        assert output.dtype == np.uint8

    def test_draw_distance_label_no_distance_value(self, sample_frame, sample_alpha):
        """Test label drawing without distance value."""
        output = draw_distance_label(sample_frame, sample_alpha, None, "Far")

        assert output.shape == sample_frame.shape
        # Text should only show label, not distance value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
