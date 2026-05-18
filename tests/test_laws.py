"""Hand-computed fixtures for the guidance laws on a two-frame chase."""

import numpy as np
import pytest

from aor.laws import (
    LAWS,
    LawParams,
    apply_delay,
    collision_course_heading,
    command_step,
    commands,
    optimal_capture_time,
    optimal_evader_heading,
)

FPS = 30.0


def two_frame_geometry():
    """Frame 0: pursuer at the origin heading +x; target at (10, 10) moving (0, -5) cm/s.
    Frame 1: pursuer has run 20 cm/s along +x for one frame; target has moved by its velocity."""
    dt = 1.0 / FPS
    px = np.array([0.0, 20.0 * dt])
    py = np.array([0.0, 0.0])
    heading = np.array([0.0, 0.0])
    tx = np.array([10.0, 10.0])
    ty = np.array([10.0, 10.0 - 5.0 * dt])
    tvx = np.array([0.0, 0.0])
    tvy = np.array([-5.0, -5.0])
    return px, py, heading, tx, ty, tvx, tvy


def test_pure_pursuit_frame0_by_hand():
    g = two_frame_geometry()
    c = commands("pure_pursuit", *g, LawParams(gain=0.5), FPS)
    # line of sight at 45 degrees, heading 0, gain one half: command = pi/8
    assert c[0] == pytest.approx(np.pi / 8, abs=1e-9)


def test_lead_pursuit_frame0_by_hand():
    g = two_frame_geometry()
    c = commands("lead_pursuit", *g, LawParams(gain=0.5, tau=1.0), FPS)
    # lead point (10, 5): angle atan(0.5) = 0.463648; half of it
    assert c[0] == pytest.approx(0.5 * 0.4636476, abs=1e-6)


def test_lead_pursuit_reduces_to_pure_pursuit_at_tau_zero():
    g = two_frame_geometry()
    a = commands("lead_pursuit", *g, LawParams(gain=0.7, tau=0.0), FPS)
    b = commands("pure_pursuit", *g, LawParams(gain=0.7), FPS)
    assert np.allclose(a, b)


def test_proportional_navigation_by_hand():
    g = two_frame_geometry()
    c = commands("proportional_navigation", *g, LawParams(N=3.0), FPS)
    assert c[0] == 0.0
    # frame 1: pursuer (0.6667, 0), target (10, 9.8333): los = atan(9.8333/9.3333) = 0.811489
    lam0 = np.pi / 4
    lam1 = np.arctan2(10.0 - 5.0 / FPS, 10.0 - 20.0 / FPS)
    assert lam1 == pytest.approx(0.811479, abs=1e-5)
    assert c[1] == pytest.approx(3.0 * (lam1 - lam0), abs=1e-9)
    assert c[1] == pytest.approx(0.078243, abs=1e-5)


def test_command_step_matches_vectorised():
    g = two_frame_geometry()
    for law, p in [
        ("pure_pursuit", LawParams(gain=0.4)),
        ("lead_pursuit", LawParams(gain=0.4, tau=0.3)),
        ("proportional_navigation", LawParams(N=2.5)),
    ]:
        vec = commands(law, *g, p, FPS)
        prev = None
        for i in range(2):
            c, prev = command_step(law, *(a[i] for a in g), p, prev)
            assert c == pytest.approx(vec[i], abs=1e-12)


def test_apply_delay_shifts_and_fills():
    c = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(apply_delay(c, 0), c)
    assert np.allclose(apply_delay(c, 2), [0.0, 0.0, 1.0, 2.0])
    assert np.all(np.isnan(apply_delay(c, 1, fill=np.nan)[:1]))


def test_unknown_law_rejected():
    g = two_frame_geometry()
    with pytest.raises(ValueError):
        commands("zigzag", *g, LawParams(), FPS)
    assert set(LAWS) == {"pure_pursuit", "lead_pursuit", "proportional_navigation"}


def test_collision_course_nulls_los_rate():
    # target ahead at (20, 0) moving +y at 6 cm/s; pursuer speed 10: heading = asin(0.6) = 0.6435
    h = collision_course_heading(0.0, 0.0, 20.0, 0.0, 0.0, 6.0, 10.0)
    assert h == pytest.approx(np.arcsin(0.6), abs=1e-9)
    # on that course the pursuer's transverse velocity equals the target's: 10 sin(h) = 6
    assert 10.0 * np.sin(h) == pytest.approx(6.0, abs=1e-9)
    # a target fleeing straight along the line of sight gives pure pursuit
    assert collision_course_heading(0.0, 0.0, 20.0, 0.0, 5.0, 0.0, 10.0) == pytest.approx(0.0)
    # transverse speed above the pursuer's speed: fall back to the line of sight
    assert collision_course_heading(0.0, 0.0, 20.0, 0.0, 0.0, 15.0, 10.0) == pytest.approx(
        np.pi / 2, abs=1e-9
    )


def test_optimal_evader_and_value():
    assert optimal_evader_heading(0.0, 0.0, 0.0, 5.0) == pytest.approx(np.pi / 2)
    assert optimal_capture_time(30.0, 20.0, 10.0, 2.0) == pytest.approx(2.8)
    assert optimal_capture_time(30.0, 10.0, 10.0, 2.0) == float("inf")
