"""Synthetic alpha experiments; these do not measure real-person matting quality."""

import numpy as np
import pytest

from cv_human_video_matting.postprocess.temporal import TemporalAlphaSmoother


def static_noise_measurements():
    """Seeded stationary soft alpha, Gaussian noise sigma=4/255, 30-frame warmup."""
    rng = np.random.default_rng(2026)
    frames = np.clip(128 + rng.normal(0, 4, (160, 48, 64)), 0, 255).astype(np.uint8)
    results = {"raw_mse": float(np.mean((frames[30:].astype(float) - 128) ** 2))}
    for mode in ("ema", "adaptive"):
        smoother = TemporalAlphaSmoother(blend=0.75, mode=mode, motion_threshold=0.15)
        outputs = np.stack([smoother.apply(frame) for frame in frames])[30:].astype(float)
        results[f"{mode}_mse"] = float(np.mean((outputs - 128) ** 2))
        results[f"{mode}_frame_delta_mse"] = float(np.mean(np.diff(outputs, axis=0) ** 2))
    return results


def moving_mask_measurements():
    """A 24x40 mask moves 3 pixels/frame with Gaussian alpha noise sigma=3/255."""
    rng = np.random.default_rng(2026)
    truth = np.zeros((32, 80, 144), dtype=np.uint8)
    for index, frame in enumerate(truth):
        left = 8 + 3 * index
        frame[20:60, left:left + 24] = 255
    inputs = np.clip(truth.astype(float) + rng.normal(0, 3, truth.shape), 0, 255).astype(np.uint8)
    vacated = (truth[:-1] == 255) & (truth[1:] == 0)
    entered = (truth[:-1] == 0) & (truth[1:] == 255)
    results = {}
    for mode in ("ema", "adaptive"):
        smoother = TemporalAlphaSmoother(blend=0.75, mode=mode, motion_threshold=0.15)
        output = np.stack([smoother.apply(frame) for frame in inputs]).astype(float)
        results[f"{mode}_mse"] = float(np.mean((output[1:] - truth[1:]) ** 2))
        results[f"{mode}_vacated_alpha"] = float(np.mean(output[1:][vacated]))
        results[f"{mode}_entered_error"] = float(np.mean(255 - output[1:][entered]))
    return results


def test_default_mode_matches_existing_ema_exactly():
    frames = np.random.default_rng(7).integers(0, 256, (20, 12, 16), dtype=np.uint8)
    legacy = TemporalAlphaSmoother(blend=0.75)
    explicit = TemporalAlphaSmoother(blend=0.75, mode="ema", motion_threshold=0.8)
    previous = None
    for frame in frames:
        current = frame.astype(np.float32)
        previous = current if previous is None else previous * 0.75 + current * 0.25
        expected = np.clip(previous, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(legacy.apply(frame), expected)
        np.testing.assert_array_equal(explicit.apply(frame), expected)


def test_adaptive_reduces_moving_boundary_trails_and_delayed_foreground():
    results = moving_mask_measurements()
    assert results["adaptive_mse"] < results["ema_mse"] * 0.1
    assert results["adaptive_vacated_alpha"] < results["ema_vacated_alpha"] * 0.1
    assert results["adaptive_entered_error"] < results["ema_entered_error"] * 0.1


def test_stationary_noise_reduction_remains_close_to_ema():
    results = static_noise_measurements()
    assert results["adaptive_mse"] <= results["ema_mse"] * 1.15
    assert results["adaptive_mse"] < results["raw_mse"] * 0.35
    assert results["adaptive_frame_delta_mse"] <= results["ema_frame_delta_mse"] * 1.25


def test_motion_is_local_and_stable_regions_keep_history():
    smoother = TemporalAlphaSmoother(blend=0.8, mode="adaptive", motion_threshold=0.2)
    first = np.full((4, 6), 100, np.uint8)
    second = first.copy()
    second[:, :2] = 200
    third = second.copy()
    third[:, -2:] = 106
    smoother.apply(first)
    moved = smoother.apply(second)
    np.testing.assert_array_equal(moved[:, :2], second[:, :2])
    np.testing.assert_array_equal(moved[:, 2:], first[:, 2:])
    noisy = smoother.apply(third)
    assert np.all(noisy[:, -2:] < third[:, -2:])
    np.testing.assert_array_equal(noisy[:, :2], second[:, :2])


@pytest.mark.parametrize("blend", [0.0, 1.0])
def test_extreme_blends_have_defined_behavior_on_large_changes(blend):
    smoother = TemporalAlphaSmoother(blend=blend, mode="adaptive", motion_threshold=0.15)
    blank = np.zeros((8, 8), np.uint8)
    full = np.full_like(blank, 255)
    np.testing.assert_array_equal(smoother.apply(blank), blank)
    np.testing.assert_array_equal(smoother.apply(full), full)
    np.testing.assert_array_equal(smoother.apply(blank), blank)


def test_zero_blend_returns_current_for_small_as_well_as_large_changes():
    smoother = TemporalAlphaSmoother(blend=0, mode="adaptive")
    for value in (100, 101, 98, 255, 0):
        frame = np.full((8, 8), value, np.uint8)
        np.testing.assert_array_equal(smoother.apply(frame), frame)


@pytest.mark.parametrize("change", ["reset", "shape"])
def test_reset_or_shape_change_discards_both_histories(change):
    smoother = TemporalAlphaSmoother(blend=0.9, mode="adaptive")
    smoother.apply(np.zeros((8, 8), np.uint8))
    smoother.apply(np.full((8, 8), 5, np.uint8))
    if change == "reset":
        smoother.reset()
        shape = (8, 8)
    else:
        shape = (6, 10)
    new_frame = np.full(shape, 10, np.uint8)
    np.testing.assert_array_equal(smoother.apply(new_frame), new_frame)
    np.testing.assert_array_equal(smoother.apply(new_frame), new_frame)


def test_normalized_float_input_matches_uint8_and_does_not_share_state():
    left = TemporalAlphaSmoother(mode="adaptive")
    right = TemporalAlphaSmoother(mode="adaptive")
    for value in (0, 32, 64, 60, 255):
        frame = np.full((5, 8), value, np.uint8)
        output = left.apply(frame)
        np.testing.assert_array_equal(output, right.apply(frame.astype(np.float32) / 255))
        frame[:] = 0
        output[:] = 0
    expected = np.full((5, 8), 255, np.uint8)
    np.testing.assert_array_equal(left.apply(expected), expected)


@pytest.mark.parametrize("mode", ["ema", "adaptive"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_alpha_is_rejected_without_poisoning_history(mode, bad_value):
    smoother = TemporalAlphaSmoother(mode=mode)
    control = TemporalAlphaSmoother(mode=mode)
    first = np.full((6, 8), 100, np.uint8)
    smoother.apply(first)
    control.apply(first)
    bad = np.full((6, 8), 0.4, np.float32)
    bad[0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        smoother.apply(bad)
    current = np.full((6, 8), 110, np.uint8)
    np.testing.assert_array_equal(smoother.apply(current), control.apply(current))


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="motion_threshold"):
        TemporalAlphaSmoother(mode="adaptive", motion_threshold=threshold)


@pytest.mark.parametrize("blend", [np.nan, np.inf, -np.inf])
def test_nonfinite_blend_is_rejected(blend):
    with pytest.raises(ValueError, match="blend"):
        TemporalAlphaSmoother(blend=blend, mode="adaptive")


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        TemporalAlphaSmoother(mode="automatic")


def test_tiny_positive_threshold_is_finite_and_responds_to_one_alpha_level():
    smoother = TemporalAlphaSmoother(mode="adaptive", motion_threshold=np.nextafter(0.0, 1.0))
    first = np.full((4, 4), 100, np.uint8)
    second = first.copy()
    second[:, 0] = 101
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(smoother.apply(first), first)
        np.testing.assert_array_equal(smoother.apply(first), first)
        np.testing.assert_array_equal(smoother.apply(second), second)
