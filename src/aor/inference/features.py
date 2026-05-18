"""Per-chase arrays for the likelihood: heading increments, the regressor of each law, the
target-manoeuvre indicator, and the transition covariates.

Alignment convention. The command computed from the geometry at frame t acts on the heading
increment from frame t + d to frame t + d + 1, where d is the motor delay in frames. A design
row for increment s therefore uses the geometry at frame s - d. Rows start at s0 >= d + 1 so
that the line-of-sight rotation (undefined at frame 0) is never used as a regressor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Chase, wrap_angle
from ..kinematics import runs_of_true
from ..laws import LAWS, los_angle


@dataclass(frozen=True)
class FeatureParams:
    lag_frames: int = 3  # compare target heading and speed against this many frames earlier
    heading_change_deg: float = 60.0  # a turn this large counts as a manoeuvre
    speed_change_cm_s: float = 15.0  # a speed change this large counts as a manoeuvre
    min_speed_cm_s: float = 3.0  # heading changes count only when the target is moving
    recent_frames: int = 6  # the covariate stays raised for this many frames after a manoeuvre
    vel_window: int = 2  # the target velocity the laws act on: backward difference over this many frames


ManoeuvreParams = FeatureParams


@dataclass
class ChaseArrays:
    chase_id: str
    animal_id: str
    fps: float
    px: np.ndarray
    py: np.ndarray
    heading: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    tvx: np.ndarray
    tvy: np.ndarray
    rng: np.ndarray
    dpsi: np.ndarray  # (T-1,) heading increments, wrapped
    bearing: np.ndarray  # (T,) pure-pursuit error: line of sight minus heading, wrapped
    dlam: np.ndarray  # (T,) line-of-sight rotation over the previous frame, 0 at frame 0
    t_speed: np.ndarray
    closing: np.ndarray  # (T,) rate of range decrease, cm/s
    manoeuvre: np.ndarray  # (T,) bool, a manoeuvre starts at this frame
    recent: np.ndarray  # (T,) float, 1 within recent_frames after a manoeuvre
    law_true: np.ndarray | None = None
    away: bool = False  # evader role: regressors point away from the other animal

    @property
    def n_frames(self) -> int:
        return len(self.px)


def manoeuvre_indicator(tvx: np.ndarray, tvy: np.ndarray, fps: float, p: FeatureParams) -> np.ndarray:
    """Frames at which the target's heading or speed changes suddenly."""
    speed = np.hypot(tvx, tvy)
    theta = np.arctan2(tvy, tvx)
    k = p.lag_frames
    n = len(speed)
    out = np.zeros(n, dtype=bool)
    if n <= k:
        return out
    dth = np.abs(wrap_angle(theta[k:] - theta[:-k]))
    dsp = np.abs(speed[k:] - speed[:-k])
    moving = (speed[k:] > p.min_speed_cm_s) & (speed[:-k] > p.min_speed_cm_s)
    hit = (moving & (dth > np.deg2rad(p.heading_change_deg))) | (dsp > p.speed_change_cm_s)
    # mark the first frame of each run of hits as the manoeuvre onset
    s, _ = runs_of_true(hit)
    out[k + s] = True
    return out


def recent_indicator(manoeuvre: np.ndarray, recent_frames: int) -> np.ndarray:
    out = np.zeros(len(manoeuvre), dtype=float)
    for i in np.flatnonzero(manoeuvre):
        out[i : i + recent_frames + 1] = 1.0
    return out


def causal_velocity(x: np.ndarray, y: np.ndarray, fps: float, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Target velocity as the pursuer could have perceived it at each frame: the backward
    difference over ``window`` frames. The first frames repeat the first available value."""
    w = max(1, int(window))
    vx = np.full_like(x, np.nan)
    vy = np.full_like(y, np.nan)
    if len(x) > w:
        vx[w:] = (x[w:] - x[:-w]) * fps / w
        vy[w:] = (y[w:] - y[:-w]) * fps / w
        vx[:w] = vx[w]
        vy[:w] = vy[w]
    else:
        vx[:] = 0.0
        vy[:] = 0.0
    return vx, vy


def prepare(chase: Chase, mp: FeatureParams = FeatureParams()) -> ChaseArrays:
    """Per-chase arrays. The manoeuvre indicator uses the table's smoothed target velocity; the
    law regressors use the causal velocity over ``mp.vel_window`` frames."""
    f = chase.frames
    px, py, heading, tx, ty, tvx_s, tvy_s = (
        f[c].to_numpy(dtype=float) for c in ("px", "py", "p_heading", "tx", "ty", "tvx", "tvy")
    )
    tvx, tvy = causal_velocity(tx, ty, chase.meta.fps, mp.vel_window)
    rng = f["range"].to_numpy(dtype=float)
    lam = los_angle(px, py, tx, ty)
    dlam = np.zeros_like(lam)
    dlam[1:] = wrap_angle(lam[1:] - lam[:-1])
    dpsi = wrap_angle(heading[1:] - heading[:-1])
    closing = np.zeros_like(rng)
    closing[1:] = -(rng[1:] - rng[:-1]) * chase.meta.fps
    man = manoeuvre_indicator(tvx_s, tvy_s, chase.meta.fps, mp)
    law_true = f["law_true"].to_numpy(dtype=float) if "law_true" in f.columns else None
    return ChaseArrays(
        chase.meta.chase_id,
        chase.meta.animal_id,
        chase.meta.fps,
        px,
        py,
        heading,
        tx,
        ty,
        tvx,
        tvy,
        rng,
        dpsi,
        f["bearing"].to_numpy(dtype=float),
        dlam,
        np.hypot(tvx_s, tvy_s),
        closing,
        man,
        recent_indicator(man, mp.recent_frames),
        law_true,
    )


def lead_error(arr: ChaseArrays, tau: float) -> np.ndarray:
    """Angle from the heading to the target's position extrapolated by tau seconds, wrapped.
    For an evader (``arr.away``) the angle points away from the extrapolated position."""
    lx = arr.tx + tau * arr.tvx
    ly = arr.ty + tau * arr.tvy
    flip = np.pi if arr.away else 0.0
    return wrap_angle(np.arctan2(ly - arr.py, lx - arr.px) + flip - arr.heading)


def prepare_evader(chase: Chase, mp: FeatureParams = FeatureParams()) -> ChaseArrays:
    """Arrays for the evader library: the target mouse is the steering animal, the chasing mouse
    is the other animal. Law 0 is pure evasion (run from the pursuer's current position), law 1
    is lead evasion (run from its extrapolated position), law 2 is straight flight (zero command).
    Requires a target heading column."""
    f = chase.frames
    if "t_heading" not in f.columns:
        raise ValueError("evader arrays need a t_heading column")
    ex, ey, eh = (f[c].to_numpy(dtype=float) for c in ("tx", "ty", "t_heading"))
    px, py = (f[c].to_numpy(dtype=float) for c in ("px", "py"))
    fps = chase.meta.fps
    pvx, pvy = causal_velocity(px, py, fps, mp.vel_window)
    p_speed = f["p_speed"].to_numpy(dtype=float)
    pvx_s, pvy_s = (
        p_speed * np.cos(f["p_heading"].to_numpy(dtype=float)),
        p_speed * np.sin(f["p_heading"].to_numpy(dtype=float)),
    )
    rng = f["range"].to_numpy(dtype=float)
    dpsi = wrap_angle(eh[1:] - eh[:-1])
    away = wrap_angle(np.arctan2(ey - py, ex - px))  # direction from pursuer to evader
    bearing = wrap_angle(away - eh)
    closing = np.zeros_like(rng)
    closing[1:] = -(rng[1:] - rng[:-1]) * fps
    man = manoeuvre_indicator(pvx_s, pvy_s, fps, mp)
    law_true = None
    return ChaseArrays(
        chase.meta.chase_id,
        chase.meta.animal_id,
        fps,
        ex,
        ey,
        eh,
        px,
        py,
        pvx,
        pvy,
        rng,
        dpsi,
        bearing,
        np.zeros_like(rng),
        p_speed,
        closing,
        man,
        recent_indicator(man, mp.recent_frames),
        law_true,
        away=True,
    )


def support(arr: ChaseArrays, delay: int, start: int | None = None) -> np.ndarray:
    """Increment indices s used in the likelihood for a given delay: s from max(start, delay + 1)
    to T - 2 inclusive."""
    s0 = max(delay + 1, start if start is not None else 0)
    return np.arange(s0, arr.n_frames - 1)


def design(
    arr: ChaseArrays, delay: int, tau: float, start: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Response y (n,), regressors X (n, 3) ordered as LAWS, and the increment indices s (n,)."""
    s = support(arr, delay, start)
    g = s - delay
    y = arr.dpsi[s]
    X = np.column_stack([arr.bearing[g], lead_error(arr, tau)[g], arr.dlam[g]])
    return y, X, s


def design_pair(arr: ChaseArrays, delay: int, tau: float, start: int | None = None):
    """Like ``design`` but also returns the response and regressors of the previous increment,
    for the autoregressive residual. Rows start at max(start, delay + 2)."""
    s0 = max(delay + 2, start if start is not None else 0)
    s = np.arange(s0, arr.n_frames - 1)
    g = s - delay
    le = lead_error(arr, tau)
    y, yp = arr.dpsi[s], arr.dpsi[s - 1]
    X = np.column_stack([arr.bearing[g], le[g], arr.dlam[g]])
    Xp = np.column_stack([arr.bearing[g - 1], le[g - 1], arr.dlam[g - 1]])
    return y, X, s, yp, Xp


def covariates(
    arr: ChaseArrays,
    s: np.ndarray,
    range_scale: float = 10.0,
    range_centre: float = 15.0,
    closing_scale: float = 10.0,
) -> np.ndarray:
    """Transition covariates at the increment indices s: intercept, scaled range, scaled closing
    speed, and the recent-manoeuvre indicator."""
    return np.column_stack(
        [
            np.ones(len(s)),
            (arr.rng[s] - range_centre) / range_scale,
            arr.closing[s] / closing_scale,
            arr.recent[s],
        ]
    )


N_LAWS = len(LAWS)
