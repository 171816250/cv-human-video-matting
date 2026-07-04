import numpy as np

from cv_human_video_matting.render.compositor import (
    alpha_to_bgr,
    compose_on_image,
    compose_on_color,
    normalize_alpha,
    resize_cover,
    side_by_side,
)


def test_normalize_alpha_accepts_uint8_mask():
    alpha = np.array([[0, 127, 255]], dtype=np.uint8)

    normalized = normalize_alpha(alpha)

    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 3, 1)
    np.testing.assert_allclose(normalized[0, :, 0], [0.0, 127 / 255, 1.0], rtol=1e-6)


def test_compose_on_color_blends_bgr_frame_with_alpha():
    frame = np.array([[[10, 20, 30], [100, 150, 200]]], dtype=np.uint8)
    alpha = np.array([[255, 0]], dtype=np.uint8)

    output = compose_on_color(frame, alpha, color_bgr=(0, 255, 0))

    assert output.dtype == np.uint8
    assert output.tolist() == [[[10, 20, 30], [0, 255, 0]]]


def test_alpha_to_bgr_returns_three_channel_visualization():
    alpha = np.array([[0, 255]], dtype=np.uint8)

    output = alpha_to_bgr(alpha)

    assert output.shape == (1, 2, 3)
    assert output.dtype == np.uint8
    assert output.tolist() == [[[0, 0, 0], [255, 255, 255]]]


def test_resize_cover_crops_without_distorting_target_size():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :10] = (10, 20, 30)
    image[:, 10:] = (100, 110, 120)

    output = resize_cover(image, size=(10, 10))

    assert output.shape == (10, 10, 3)


def test_compose_on_image_uses_resized_background():
    frame = np.array([[[10, 20, 30], [100, 150, 200]]], dtype=np.uint8)
    alpha = np.array([[255, 0]], dtype=np.uint8)
    background = np.full((4, 4, 3), (5, 6, 7), dtype=np.uint8)

    output = compose_on_image(frame, alpha, background)

    assert output.dtype == np.uint8
    assert output.tolist() == [[[10, 20, 30], [5, 6, 7]]]


def test_side_by_side_concatenates_matching_height_frames():
    left = np.zeros((2, 3, 3), dtype=np.uint8)
    right = np.full((2, 3, 3), 255, dtype=np.uint8)

    output = side_by_side(left, right)

    assert output.shape == (2, 6, 3)
    assert output[:, :3].sum() == 0
    assert output[:, 3:].sum() == 2 * 3 * 3 * 255
