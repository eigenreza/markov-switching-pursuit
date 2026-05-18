import numpy as np
import pytest

from aor.kinematics import (
    angular_difference,
    despike,
    heading_from_axis,
    heading_from_velocity,
    interpolate_gaps,
    moving_average,
    repair_flips,
    runs_of_true,
    speed,
    velocity,
)


def test_moving_average_ignores_nan_and_keeps_length():
    x = np.array([1.0, np.nan, 3.0, 5.0, np.nan, np.nan, np.nan, 9.0])
    y = moving_average(x, 3, min_valid_fraction=0.5)
    assert y.shape == x.shape
    assert y[1] == pytest.approx(2.0)  # mean of 1 and 3
    assert np.isnan(y[5])  # window [nan, nan, nan] -> 0 valid of 3


def test_velocity_of_linear_motion():
    fps = 30.0
    t = np.arange(90) / fps
    xy = np.column_stack([3.0 * t, -2.0 * t])
    v = velocity(xy, fps)
    assert np.allclose(v[:, 0], 3.0) and np.allclose(v[:, 1], -2.0)
    assert np.allclose(speed(xy, fps), np.hypot(3.0, 2.0))


def test_heading_from_axis_points_forward():
    rear = np.zeros((5, 2))
    front = np.tile([0.0, 1.0], (5, 1))  # pointing +y
    h = heading_from_axis(rear, front)
    assert np.allclose(h, np.pi / 2)
    front = np.tile([-1.0, 0.0], (5, 1))  # pointing -x -> pi (wrapped into (-pi, pi])
    assert np.allclose(np.abs(heading_from_axis(rear, front)), np.pi)


def test_heading_from_velocity_masks_slow_frames():
    fps = 30.0
    xy = np.column_stack([np.linspace(0, 10, 31), np.zeros(31)])
    h = heading_from_velocity(xy, fps, min_speed=5.0)
    assert np.allclose(h, 0.0)
    h2 = heading_from_velocity(xy, fps, min_speed=50.0)
    assert np.all(np.isnan(h2))


def test_angular_difference_wraps():
    assert angular_difference(np.pi - 0.1, -np.pi + 0.1) == pytest.approx(-0.2)


def test_interpolate_short_gaps_only():
    x = np.array([0.0, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0])
    y = interpolate_gaps(x, max_gap=1)
    assert y[1] == pytest.approx(1.0)
    assert np.isnan(y[3]) and np.isnan(y[5])
    z = interpolate_gaps(x, max_gap=3)
    assert np.allclose(z, np.arange(7, dtype=float))


def test_runs_of_true():
    s, e = runs_of_true(np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool))
    assert s.tolist() == [1, 4, 7] and e.tolist() == [3, 5, 8]


def test_despike_removes_single_jump():
    fps = 30.0
    xy = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
    xy[4] = [50.0, 50.0]
    cleaned = despike(xy, fps, max_speed=100.0)
    assert np.isnan(cleaned[4]).all()
    assert np.isfinite(cleaned[5]).all()


def test_repair_flips_replaces_single_frame_flips_only():
    h = np.linspace(0.0, 1.0, 12)
    h[5] += np.pi  # a single-frame flip
    h[8] += 0.4  # a real turn that stays
    out = repair_flips(h, threshold=np.pi / 2)
    assert abs(out[5] - 0.5 * (h[4] + h[6])) < 1e-9
    assert out[8] == h[8]
    keep = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11]
    assert np.all(out[keep] == h[keep])
    h[6] = np.nan
    out = repair_flips(h)
    assert out[5] == h[5]  # a flip next to a gap is left alone
