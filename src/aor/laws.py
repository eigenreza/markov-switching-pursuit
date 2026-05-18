"""The guidance-law library and the optimal-play reference strategies.

Each law maps the geometry at one frame (pursuer position and heading, target position and
velocity) to a commanded heading change for that frame, in radians per frame. The observed
heading change is the command delayed by a fixed number of frames plus noise; the delay and
the noise are handled by the simulator and the inference, not here.

Laws
- pure_pursuit: steer toward the target's current position. Command = gain * (line-of-sight
  angle - heading), wrapped.
- lead_pursuit: steer toward the target's position extrapolated by a lead time tau along its
  current velocity. Command = gain * (angle to the lead point - heading). Pure pursuit is the
  case tau = 0.
- proportional_navigation: turn in proportion to the rotation of the line of sight.
  Command = N * (change in line-of-sight angle over the frame). The gain parameter is not used
  by this law; N takes its place as the coefficient. Large N drives the line-of-sight rate to
  zero, which is parallel navigation (constant bearing).

Reference strategies of the simple-motion pursuit game (constant speeds, free headings, the
pursuer faster): the pursuer's collision-course heading, which holds the line of sight at a
constant bearing given the target's current velocity, and the evader's straight flight along
the line of sight away from the pursuer. See docs/methods.md for the statement and sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import wrap_angle

LAWS = ("pure_pursuit", "lead_pursuit", "proportional_navigation")
LAW_INDEX = {name: i for i, name in enumerate(LAWS)}
ANTICIPATORY = {"pure_pursuit": False, "lead_pursuit": True, "proportional_navigation": True}


@dataclass(frozen=True)
class LawParams:
    """Parameters of one law. ``gain`` is the fraction of the heading error corrected per frame
    for the pursuit laws; ``N`` is the navigation constant of proportional navigation; ``tau``
    is the lead time in seconds; ``delay`` is the motor delay in frames; ``noise`` is the standard
    deviation of the heading-increment noise in radians per frame."""

    gain: float = 0.5
    delay: int = 0
    noise: float = 0.05
    tau: float = 0.0
    N: float = 3.0
    noise_ar: float = 0.0  # first-order autocorrelation of the heading-increment noise


def los_angle(px, py, tx, ty):
    """Line-of-sight angle from pursuer to target, radians in (-pi, pi]."""
    return wrap_angle(np.arctan2(np.asarray(ty) - np.asarray(py), np.asarray(tx) - np.asarray(px)))


def lead_point(tx, ty, tvx, tvy, tau: float):
    """Target position extrapolated by tau seconds along its current velocity."""
    return np.asarray(tx) + tau * np.asarray(tvx), np.asarray(ty) + tau * np.asarray(tvy)


def desired_heading(law: str, px, py, heading, tx, ty, tvx, tvy, params: LawParams):
    """Desired heading for the pursuit laws (not defined for proportional navigation)."""
    if law == "pure_pursuit":
        return los_angle(px, py, tx, ty)
    if law == "lead_pursuit":
        lx, ly = lead_point(tx, ty, tvx, tvy, params.tau)
        return los_angle(px, py, lx, ly)
    raise ValueError(f"desired_heading is not defined for {law!r}")


def commands(law: str, px, py, heading, tx, ty, tvx, tvy, params: LawParams, fps: float) -> np.ndarray:
    """Commanded heading change (radians per frame) at every frame of a chase, computed from
    the geometry of that same frame. Proportional navigation needs the previous frame for the
    line-of-sight rate, so its command at frame 0 is zero.

    Note: the command is the law's output before any delay; callers shift it by the delay."""
    px, py, heading, tx, ty, tvx, tvy = (
        np.asarray(a, dtype=float) for a in (px, py, heading, tx, ty, tvx, tvy)
    )
    if law in ("pure_pursuit", "lead_pursuit"):
        goal = desired_heading(law, px, py, heading, tx, ty, tvx, tvy, params)
        return params.gain * wrap_angle(goal - heading)
    if law == "proportional_navigation":
        lam = los_angle(px, py, tx, ty)
        dlam = np.zeros_like(lam)
        dlam[1:] = wrap_angle(lam[1:] - lam[:-1])
        return params.N * dlam
    raise ValueError(f"unknown law {law!r}")


def command_step(law: str, px, py, heading, tx, ty, tvx, tvy, params: LawParams, prev_los: float | None):
    """One-step version for the simulator. Returns (command, current line-of-sight angle)."""
    lam = float(los_angle(px, py, tx, ty))
    if law in ("pure_pursuit", "lead_pursuit"):
        goal = float(desired_heading(law, px, py, heading, tx, ty, tvx, tvy, params))
        return params.gain * float(wrap_angle(goal - heading)), lam
    if law == "proportional_navigation":
        dlam = 0.0 if prev_los is None else float(wrap_angle(lam - prev_los))
        return params.N * dlam, lam
    raise ValueError(f"unknown law {law!r}")


def apply_delay(cmd: np.ndarray, delay: int, fill: float = 0.0) -> np.ndarray:
    """Shift commands by ``delay`` frames: the command computed at frame t acts at frame t + delay.
    The first ``delay`` entries are filled with ``fill``."""
    cmd = np.asarray(cmd, dtype=float)
    if delay <= 0:
        return cmd.copy()
    out = np.full_like(cmd, fill)
    out[delay:] = cmd[:-delay]
    return out


def collision_course_heading(px, py, tx, ty, tvx, tvy, pursuer_speed):
    """Heading that nulls the line-of-sight rate given the target's current velocity: the
    constant-bearing (parallel navigation) intercept course. Where the target's transverse speed
    exceeds the pursuer's speed no such course exists and the line-of-sight angle is returned.

    With line-of-sight angle lam, target velocity (tvx, tvy) and pursuer speed v, the course is
    psi = lam + asin(v_t_perp / v), where v_t_perp is the target's velocity component
    perpendicular to the line of sight."""
    lam = los_angle(px, py, tx, ty)
    tvx, tvy, v = (np.asarray(a, dtype=float) for a in (tvx, tvy, pursuer_speed))
    v_perp = -tvx * np.sin(lam) + tvy * np.cos(lam)
    ratio = np.divide(v_perp, v, out=np.zeros_like(v_perp, dtype=float), where=v > 0)
    ratio = np.clip(ratio, -1.0, 1.0)
    return wrap_angle(lam + np.arcsin(ratio))


def optimal_evader_heading(px, py, tx, ty):
    """The evader's optimal strategy in the simple-motion game: run straight along the line of
    sight away from the pursuer."""
    return los_angle(px, py, tx, ty)


def optimal_capture_time(
    range0: float, pursuer_speed: float, evader_speed: float, capture_radius: float
) -> float:
    """Value of the simple-motion pursuit game: time to capture under optimal play by both sides,
    (range - capture radius) / (pursuer speed - evader speed). Infinite if the pursuer is not faster."""
    if pursuer_speed <= evader_speed:
        return float("inf")
    return max(0.0, (range0 - capture_radius)) / (pursuer_speed - evader_speed)
