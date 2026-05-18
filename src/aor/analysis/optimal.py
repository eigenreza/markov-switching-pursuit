"""The optimal-play benchmark on real chases.

Distance from optimal play: the time average, over a chase, of the absolute wrapped difference
between the heading the pursuer actually adopted and the constant-bearing collision course that
the simple-motion game prescribes from the same geometry (position, target velocity as the
pursuer could perceive it, and the pursuer's own speed). With a motor delay d the heading
adopted at frame s + d + 1 is compared with the course prescribed at frame s. A moving-block
bootstrap over frames gives a per-chase interval.

Optimal capture time: the time at which a pursuer running the collision course without delay,
at the recorded speed profile and from the recorded start, first comes within the contact
radius of the recorded target path. It is compared with the recorded capture time for chases
that ended in contact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import Chase, wrap_angle
from ..inference.features import causal_velocity
from ..laws import LawParams, collision_course_heading
from ..simulator import simulate_along_target
from .datasets import informative_mask


def distance_from_optimal(
    chase: Chase,
    delay: int,
    vel_window: int = 2,
    v_min: float = 3.0,
    B: int = 300,
    block: int = 10,
    seed: int = 0,
) -> dict:
    f = chase.frames
    px, py, heading, tx, ty, speed = (
        f[c].to_numpy(dtype=float) for c in ("px", "py", "p_heading", "tx", "ty", "p_speed")
    )
    fps = chase.meta.fps
    tvx, tvy = causal_velocity(tx, ty, fps, vel_window)
    n = len(f)
    s = np.arange(0, n - delay - 1)
    if len(s) < 2:
        return {"chase_id": chase.meta.chase_id, "n_frames": 0}
    psi_star = collision_course_heading(
        px[s], py[s], tx[s], ty[s], tvx[s], tvy[s], np.maximum(speed[s], 1e-6)
    )
    adopted = heading[s + delay + 1]
    gap = np.abs(wrap_angle(adopted - psi_star))
    # the same quantity against pure pursuit, for reference
    los = np.arctan2(ty[s] - py[s], tx[s] - px[s])
    gap_pure = np.abs(wrap_angle(adopted - los))
    inf = informative_mask(chase, v_min)[s]
    rng = np.random.default_rng(seed)
    means = []
    nb = max(1, int(np.ceil(len(gap) / block)))
    for _ in range(B):
        starts = rng.integers(0, max(1, len(gap) - block + 1), nb)
        idx = np.concatenate([np.arange(st, min(st + block, len(gap))) for st in starts])[: len(gap)]
        means.append(gap[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "chase_id": chase.meta.chase_id,
        "animal_id": chase.meta.animal_id,
        "n_frames": len(gap),
        "n_informative": int(inf.sum()),
        "distance_rad": float(gap.mean()),
        "distance_lo": float(lo),
        "distance_hi": float(hi),
        "distance_informative_rad": float(gap[inf].mean()) if inf.any() else np.nan,
        "distance_to_pure_pursuit_rad": float(gap_pure.mean()),
        "mean_range": float(f["range"].mean()),
        "mean_t_speed": float(f["t_speed"].mean()),
        "mean_p_speed": float(speed.mean()),
        "duration_s": chase.duration_s,
        "end_reason": chase.meta.end_reason,
    }


def optimal_capture(chase: Chase, contact_radius: float = 3.0, vel_window: int = 2) -> dict:
    f = chase.frames
    tx, ty = f["tx"].to_numpy(dtype=float), f["ty"].to_numpy(dtype=float)
    res = simulate_along_target(
        tx,
        ty,
        chase.meta.fps,
        float(f["px"].iloc[0]),
        float(f["py"].iloc[0]),
        float(f["p_heading"].iloc[0]),
        f["p_speed"].to_numpy(dtype=float),
        "pure_pursuit",
        LawParams(gain=1.0, delay=0, noise=0.0),
        seed=0,
        capture_radius=contact_radius,
        vel_window=vel_window,
        collision_course=True,
    )
    cap = res["capture_frame"]
    actual = chase.duration_s if chase.meta.end_reason == "contact" else np.nan
    return {
        "chase_id": chase.meta.chase_id,
        "animal_id": chase.meta.animal_id,
        "optimal_capture_s": (cap / chase.meta.fps) if cap is not None else np.nan,
        "optimal_captured_within_chase": cap is not None,
        "actual_capture_s": actual,
        "end_reason": chase.meta.end_reason,
        "duration_s": chase.duration_s,
        "start_range": float(f["range"].iloc[0]),
        "end_range": float(f["range"].iloc[-1]),
    }


def phase_profile(chases: list[Chase], delay: int, vel_window: int = 2, n_bins: int = 5) -> pd.DataFrame:
    """Distance from optimal play by phase of the chase (fifths of normalised time), by range bin,
    and by target-speed bin, pooled over frames."""
    rows = []
    for c in chases:
        f = c.frames
        px, py, heading, tx, ty, speed = (
            f[k].to_numpy(dtype=float) for k in ("px", "py", "p_heading", "tx", "ty", "p_speed")
        )
        tvx, tvy = causal_velocity(tx, ty, c.meta.fps, vel_window)
        n = len(f)
        s = np.arange(0, n - delay - 1)
        if len(s) < 2:
            continue
        psi = collision_course_heading(px[s], py[s], tx[s], ty[s], tvx[s], tvy[s], np.maximum(speed[s], 1e-6))
        gap = np.abs(wrap_angle(heading[s + delay + 1] - psi))
        rows.append(
            pd.DataFrame(
                {
                    "chase_id": c.meta.chase_id,
                    "phase": s / max(len(s) - 1, 1),
                    "range": f["range"].to_numpy()[s],
                    "t_speed": f["t_speed"].to_numpy()[s],
                    "gap": gap,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    out = []
    for var, edges in (
        ("phase", np.linspace(0, 1, n_bins + 1)),
        ("range", np.quantile(df["range"], np.linspace(0, 1, n_bins + 1))),
        ("t_speed", np.quantile(df["t_speed"], np.linspace(0, 1, n_bins + 1))),
    ):
        b = np.clip(np.digitize(df[var], edges[1:-1]), 0, n_bins - 1)
        for k in range(n_bins):
            sel = b == k
            out.append(
                {
                    "variable": var,
                    "bin": k,
                    "lo": float(edges[k]),
                    "hi": float(edges[k + 1]),
                    "gap_mean_rad": float(df.loc[sel, "gap"].mean()),
                    "n_frames": int(sel.sum()),
                }
            )
    return pd.DataFrame(out)


def run(
    chases: list[Chase],
    delay: int,
    vel_window: int = 2,
    v_min: float = 3.0,
    contact_radius: float = 3.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist = pd.DataFrame(
        [distance_from_optimal(c, delay, vel_window, v_min, seed=seed + i) for i, c in enumerate(chases)]
    )
    cap = pd.DataFrame([optimal_capture(c, contact_radius, vel_window) for c in chases])
    prof = phase_profile(chases, delay, vel_window)
    return dist, cap, prof
