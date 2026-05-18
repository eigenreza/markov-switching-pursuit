"""The two-player test: does the target respond to the pursuer's current mode?

1. A logistic regression of target-manoeuvre onsets, frame by frame, on range, closing speed and
   the pursuer's posterior steering probability at the previous frame, with chase-level
   cluster bootstrap intervals and a null that circularly shifts the steering covariate within
   each chase. The shift keeps each chase's mean of the covariate, so the null distribution
   carries the between-chase association and the test statistic is the distance of the observed
   coefficient from the centre of that distribution: the within-chase effect.
2. For mouse-and-mouse chases, the evader library fitted to the target mouse's steering (pure
   evasion, lead evasion, straight flight), the cross-correlation of the two animals' posterior
   mode probabilities over time, with a chase-shuffle null, and the joint occupancy of pursuer
   mode by evader mode, overall and by phase of the chase.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..contracts import Chase
from ..inference.features import FeatureParams, prepare, prepare_evader
from ..inference.models import FitSettings, fit_switching, posteriors

EVADER_LAWS = ["pure_evasion", "lead_evasion", "straight_flight"]


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, n_iter: int = 50) -> np.ndarray:
    """Maximum-likelihood logistic regression by Newton steps with a small ridge."""
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (p - y) + l2 * w
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w = w - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def manoeuvre_rows(chases: list[Chase], frames_post: pd.DataFrame, mp: FeatureParams) -> pd.DataFrame:
    post_by = {cid: g.set_index("frame") for cid, g in frames_post.groupby("chase_id")}
    rows = []
    for c in chases:
        if c.meta.chase_id not in post_by:
            continue
        arr = prepare(c, mp)
        g = post_by[c.meta.chase_id]
        s = g.index.to_numpy()
        # the filtered (causal) state at the previous frame; the smoothed one would use frames
        # after the manoeuvre and leak the target's move into the pursuer's state
        steer = g["p_steering_filtered" if "p_steering_filtered" in g.columns else "p_steering"].to_numpy()
        spath = g["steering_filtered" if "steering_filtered" in g.columns else "steering_path"].to_numpy()
        # the pursuer's state at the previous increment predicts a manoeuvre onset at frame s
        prev = pd.Series(steer, index=s).reindex(s - 1).to_numpy()
        prev_path = pd.Series(spath, index=s).reindex(s - 1).to_numpy()
        ok = np.isfinite(prev)
        rows.append(
            pd.DataFrame(
                {
                    "chase_id": c.meta.chase_id,
                    "frame": s[ok],
                    "manoeuvre": arr.manoeuvre[s[ok]].astype(int),
                    "range": arr.rng[s[ok]],
                    "closing": arr.closing[s[ok]],
                    "p_steering_prev": prev[ok],
                    "steering_prev": prev_path[ok].astype(int),
                    "t_speed": arr.t_speed[s[ok]],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def manoeuvre_test(rows: pd.DataFrame, B: int = 500, n_perm: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y = rows["manoeuvre"].to_numpy(dtype=float)
    X = np.column_stack(
        [
            np.ones(len(rows)),
            (rows["range"] - 15.0) / 10.0,
            rows["closing"] / 10.0,
            rows["p_steering_prev"],
        ]
    )
    names = ["intercept", "range_per_10cm", "closing_per_10cm_s", "p_steering_prev"]
    w = logistic_fit(X, y)
    chase_ids = rows["chase_id"].to_numpy()
    uniq = np.unique(chase_ids)
    idx_by = {cid: np.flatnonzero(chase_ids == cid) for cid in uniq}
    boots = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        boots.append(logistic_fit(X[idx], y[idx]))
    boots = np.array(boots)
    perm = []
    for _ in range(n_perm):
        Xp = X.copy()
        for cid in uniq:
            idx = idx_by[cid]
            if len(idx) > 1:
                k = rng.integers(1, len(idx))
                Xp[idx, 3] = np.roll(X[idx, 3], k)
        perm.append(logistic_fit(Xp, y)[3])
    perm = np.array(perm)
    coef = {
        n: {
            "estimate": float(w[i]),
            "lo": float(np.percentile(boots[:, i], 2.5)),
            "hi": float(np.percentile(boots[:, i], 97.5)),
        }
        for i, n in enumerate(names)
    }
    centre = float(perm.mean())
    p_perm = float((np.sum(np.abs(perm - centre) >= abs(w[3] - centre)) + 1) / (len(perm) + 1))
    by_mode = rows.groupby("steering_prev")["manoeuvre"].agg(["mean", "size"]).reset_index()
    return {
        "coefficients": coef,
        "permutation_p_steering": p_perm,
        "permutation_null_mean": centre,
        "permutation_null_sd": float(perm.std()),
        "within_chase_effect": float(w[3] - centre),
        "n_frames": len(rows),
        "n_chases": len(uniq),
        "n_manoeuvres": int(y.sum()),
        "manoeuvre_rate_per_frame_by_pursuer_state": {
            ("steering" if int(r["steering_prev"]) else "non-steering"): {
                "rate": float(r["mean"]),
                "n_frames": int(r["size"]),
            }
            for _, r in by_mode.iterrows()
        },
    }


def evader_posteriors(
    chases: list[Chase], settings: FitSettings, mp: FeatureParams, delay: int
) -> tuple[pd.DataFrame, dict]:
    """Fit the evader library to the target mice and return per-frame posteriors."""
    arrs = [prepare_evader(c, mp) for c in chases]
    st = replace(settings, delay=delay, n_restarts=1)
    fit = fit_switching(arrs, st, transitions="plain")
    post = posteriors(arrs, fit, mp)
    frames = []
    for p in post:
        g = p["gamma"]
        frames.append(
            pd.DataFrame(
                {
                    "chase_id": p["chase_id"],
                    "frame": p["s"],
                    "e_pure": g[:, 0],
                    "e_lead": g[:, 1],
                    "e_straight": g[:, 2],
                    "e_path": p["path"],
                }
            )
        )
    summary = fit.summary()
    summary["laws"] = EVADER_LAWS
    return pd.concat(frames, ignore_index=True), summary


def cross_correlation(
    pursuer: pd.DataFrame, evader: pd.DataFrame, max_lag: int = 10, n_shuffle: int = 300, seed: int = 0
) -> pd.DataFrame:
    """Cross-correlation between the pursuer's steering probability and the evader's lead-evasion
    probability at lags in frames (positive lag: evader follows pursuer), pooled over chases after
    removing each chase's mean, with a chase-shuffle null band."""
    rng = np.random.default_rng(seed)
    m = pursuer.merge(evader, on=["chase_id", "frame"])
    series = {}
    for cid, g in m.groupby("chase_id"):
        x = g["p_steering"].to_numpy()
        y = g["e_lead"].to_numpy()
        if len(x) > 2 * max_lag + 2:
            series[cid] = (x - x.mean(), y - y.mean())
    ids = list(series)
    lags = np.arange(-max_lag, max_lag + 1)

    def pooled(pairs):
        out = []
        for lag in lags:
            num = den_x = den_y = 0.0
            for x0, y0 in pairs:
                n = min(len(x0), len(y0))
                x, y = x0[:n], y0[:n]
                if lag >= 0:
                    xs, ys = x[: n - lag], y[lag:]
                else:
                    xs, ys = x[-lag:], y[: n + lag]
                num += float(np.sum(xs * ys))
                den_x += float(np.sum(xs * xs))
                den_y += float(np.sum(ys * ys))
            out.append(num / np.sqrt(den_x * den_y) if den_x > 0 and den_y > 0 else np.nan)
        return np.array(out)

    obs = pooled([series[i] for i in ids])
    null = []
    for _ in range(n_shuffle):
        perm = rng.permutation(len(ids))
        null.append(pooled([(series[ids[i]][0], series[ids[perm[i]]][1]) for i in range(len(ids))]))
    null = np.array(null)
    return pd.DataFrame(
        {
            "lag_frames": lags,
            "correlation": obs,
            "null_lo": np.nanpercentile(null, 2.5, axis=0),
            "null_hi": np.nanpercentile(null, 97.5, axis=0),
            "n_chases": len(ids),
        }
    )


def joint_occupancy(pursuer: pd.DataFrame, evader: pd.DataFrame, n_phase: int = 5) -> pd.DataFrame:
    """Two-by-two occupancy of pursuer state (steering or non-steering) by evader mode (pure evasion
    or other), overall and by fifth of normalised chase time."""
    m = pursuer.merge(evader, on=["chase_id", "frame"])
    m["pursuer_mode"] = np.where(m["steering_path"] == 1, "steering", "non-steering")
    m["evader_mode"] = np.where(m["e_path"] == 0, "pure_evasion", "other")
    m["phase"] = m.groupby("chase_id")["frame"].transform(lambda s: (s - s.min()) / max(s.max() - s.min(), 1))
    m["phase_bin"] = np.clip((m["phase"] * n_phase).astype(int), 0, n_phase - 1)
    rows = []
    overall = m.groupby(["pursuer_mode", "evader_mode"]).size()
    for (pm, em), n in overall.items():
        rows.append(
            {
                "phase_bin": -1,
                "pursuer_mode": pm,
                "evader_mode": em,
                "fraction": n / len(m),
                "n_frames": int(n),
            }
        )
    for b, g in m.groupby("phase_bin"):
        cnt = g.groupby(["pursuer_mode", "evader_mode"]).size()
        for (pm, em), n in cnt.items():
            rows.append(
                {
                    "phase_bin": int(b),
                    "pursuer_mode": pm,
                    "evader_mode": em,
                    "fraction": n / len(g),
                    "n_frames": int(n),
                }
            )
    return pd.DataFrame(rows)
