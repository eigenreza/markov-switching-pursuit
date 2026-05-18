"""The real-data fitting pipeline.

For a dataset with a development split and a held-out split: profile the delay and fit every
model on the development chases; score every model on the held-out chases once, with chase-level
bootstrap intervals of the per-frame differences; build the switching null by parametric
bootstrap (the best single law replayed along the recorded target paths with tracking noise);
compute posteriors on the held-out chases from the development parameters; summarise them per
chase and per animal as the steering probability (the posterior mass of the states whose fitted
response to the target is non-zero); align the steering probability to target manoeuvres; and
fit the lead time and the navigation constant within chase-level bins of target speed and range.

For a replication set with no split, the same steps run in-sample with the frozen structure. The
pipeline also serves the post-freeze follow-up with a persistence state (``persistence=True``),
the scale sensitivity of the primary dataset (``variant``), and a leave-one-animal-out table.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..contracts import Chase, ChaseMeta, build_frames
from ..inference.comparator import fit_switching_ar, score_frames_switching_ar, score_switching_ar
from ..inference.features import FeatureParams, prepare
from ..inference.models import (
    PURE,
    Emissions,
    Fit,
    FitSettings,
    _designs,
    arrays,
    best_delay,
    bootstrap,
    delay_profile,
    fit_null,
    fit_single,
    fit_switching,
    interval,
    posteriors,
    score,
    score_frames,
    state_roles,
    steering_probability,
)
from ..kinematics import speed as speed_of
from ..kinematics import velocity
from ..laws import LAWS, LawParams
from ..simulator import simulate_along_target
from .datasets import DERIVED_DIR, INFORMATIVE_SPEED, informative_mask, load_real, summarise_split

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "model.yaml"
RESULTS = ROOT / "results" / "fits"
RESULTS_FOLLOWUP = ROOT / "results" / "fits_persistence"
STATE_NAMES = ["pure", "lead", "pn", "persistence"]


def load_cfg(config_path: Path = CONFIG) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def settings_from_cfg(cfg: dict) -> tuple[FitSettings, FitSettings, list[int]]:
    inf = cfg["inference"]
    mp = FeatureParams(**inf.get("manoeuvre", {}))
    full = FitSettings(
        tau_bounds=tuple(inf["tau_bounds"]),
        n_iter=int(inf["n_iter"]),
        tol=float(inf["tol"]),
        n_restarts=int(inf["n_restarts"]),
        l2=float(inf["l2"]),
        manoeuvre=mp,
    )
    light = replace(full, n_iter=int(inf["profile_n_iter"]), n_restarts=int(inf["profile_n_restarts"]))
    return full, light, list(inf["delays"])


def _per_dataset(block, dataset: str, default):
    """A config entry that is either a scalar or {default: x, <dataset>: y}."""
    if isinstance(block, dict):
        return block.get(dataset, block.get("default", default))
    return default if block is None else block


# ----------------------------------------------------------------------------- helpers


def chase_strata(chases: list[Chase]) -> list[str]:
    return [c.meta.animal_id for c in chases]


def best_single(fits: dict) -> str:
    return max((k for k in fits if k.startswith("single")), key=lambda k: fits[k].loglik)


def per_chase_scores(chases: list[Chase], fits: dict, ar_fit, delay: int, mp: FeatureParams) -> pd.DataFrame:
    start = delay + 2
    cols = {}
    for name, f in fits.items():
        sc = score(chases, f, start=start, mp=mp)
        cols[name] = sc["per_chase"]
    if ar_fit is not None:
        cols["switching_ar"] = score_switching_ar(chases, ar_fit, start=start, mp=mp)["per_chase"]
    ids = [c.meta.chase_id for c in chases if c.meta.chase_id in cols["null"]]
    animal = {c.meta.chase_id: c.meta.animal_id for c in chases}
    rows = []
    for cid in ids:
        n = cols["null"][cid][1]
        row = {"chase_id": cid, "animal_id": animal[cid], "n_frames": n}
        for name, per in cols.items():
            if cid in per and per[cid][1] == n:
                row[name] = per[cid][0]
            else:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def masked_scores(
    chases: list[Chase], fits: dict, ar_fit, delay: int, mp: FeatureParams, v_min: float
) -> pd.DataFrame:
    """Per-chase log-likelihood of each model summed over informative frames only (target moving
    faster than v_min), on the common support."""
    start = delay + 2
    per = {name: score_frames(chases, f, start=start, mp=mp) for name, f in fits.items()}
    if ar_fit is not None:
        per["switching_ar"] = score_frames_switching_ar(chases, ar_fit, start=start, mp=mp)
    rows = []
    for c in chases:
        cid = c.meta.chase_id
        if cid not in per["null"]:
            continue
        s, _ = per["null"][cid]
        mask = informative_mask(c, v_min)[s]
        if mask.sum() == 0:
            continue
        row = {"chase_id": cid, "animal_id": c.meta.animal_id, "n_frames": int(mask.sum())}
        for name, d in per.items():
            if cid in d and len(d[cid][1]) == len(mask):
                row[name] = float(d[cid][1][mask].sum())
            else:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(scores: pd.DataFrame, models: list[str], B: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per-frame log-likelihood of each model and differences against the best single law, with
    chase-level bootstrap intervals (resampling chases within animals)."""
    rng = np.random.default_rng(seed)
    sc = scores.dropna(subset=models)
    n = sc["n_frames"].to_numpy()
    groups = {a: np.flatnonzero(sc["animal_id"].to_numpy() == a) for a in sc["animal_id"].unique()}
    singles = [m for m in models if m.startswith("single")]
    best = max(singles, key=lambda m: sc[m].sum())
    rows = []
    for m in models:
        ll = sc[m].to_numpy()
        diff = ll - sc[best].to_numpy()
        pf, df = [], []
        for _ in range(B):
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups.values()])
            pf.append(ll[idx].sum() / n[idx].sum())
            df.append(diff[idx].sum() / n[idx].sum())
        lo, hi = interval(pf)
        dlo, dhi = interval(df)
        rows.append(
            {
                "model": m,
                "per_frame": ll.sum() / n.sum(),
                "per_frame_lo": lo,
                "per_frame_hi": hi,
                "per_chase": ll.mean(),
                "diff_vs_best_single_per_frame": diff.sum() / n.sum(),
                "diff_lo": dlo,
                "diff_hi": dhi,
                "best_single": best,
                "n_chases": len(sc),
                "n_frames": int(n.sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_models(dev: list[Chase], st: FitSettings, mp: FeatureParams) -> tuple[dict, object]:
    """The null, the three single laws, the two switching models and the comparator on one set."""
    fits = {"null": fit_null(dev, st)}
    for law in LAWS:
        fits[f"single:{law}"] = fit_single(dev, law, st)
    fits["switching:plain"] = fit_switching(dev, st, transitions="plain")
    fits["switching:input"] = fit_switching(
        dev,
        replace(st, n_restarts=1),
        transitions="input",
        init=fits["switching:plain"].emissions,
        init_A=fits["switching:plain"].A,
    )
    ar = fit_switching_ar(dev, st.delay, mp=mp, settings=st, n_states=3 + int(st.persistence))
    return fits, ar


# ----------------------------------------------------------------------------- parametric null


def replay_dataset(
    chases: list[Chase],
    law: str,
    em: Emissions,
    delay: int,
    seed: int,
    vel_window: int,
    obs_pos_sd: float,
    obs_heading_sd: float,
) -> list[Chase]:
    """Replace every chase's pursuer by a simulated one following ``law`` along the recorded target
    path, with the recorded speed profile and start, plus tracking noise."""
    out = []
    params = LawParams(gain=em.gain, delay=delay, noise=em.sigma, tau=em.tau, N=em.N)
    for i, c in enumerate(chases):
        f = c.frames
        n = len(f)
        res = simulate_along_target(
            f["tx"].to_numpy(),
            f["ty"].to_numpy(),
            c.meta.fps,
            float(f["px"].iloc[0]),
            float(f["py"].iloc[0]),
            float(f["p_heading"].iloc[0]),
            f["p_speed"].to_numpy(),
            law,
            params,
            seed=seed * 100003 + i,
            vel_window=vel_window,
            obs_pos_sd=obs_pos_sd,
            obs_heading_sd=obs_heading_sd,
        )
        o = (
            res["obs"]
            if res["obs"]
            else {
                "px": res["px"],
                "py": res["py"],
                "tx": f["tx"].to_numpy(),
                "ty": f["ty"].to_numpy(),
                "heading": res["heading"],
            }
        )
        pxy = np.column_stack([o["px"], o["py"]])
        txy = np.column_stack([o["tx"], o["ty"]])
        tv = velocity(txy, c.meta.fps, window=5)
        df = build_frames(
            np.arange(n),
            c.meta.fps,
            o["px"],
            o["py"],
            o["heading"],
            speed_of(pxy, c.meta.fps, window=5),
            o["tx"],
            o["ty"],
            tv[:, 0],
            tv[:, 1],
            t_speed=np.hypot(tv[:, 0], tv[:, 1]),
        )
        meta = ChaseMeta(
            "null",
            c.meta.session_id,
            c.meta.animal_id,
            c.meta.chase_id,
            c.meta.fps,
            0,
            n - 1,
            "clip_end",
            "simulated",
            native_fps=c.meta.fps,
        )
        out.append(Chase(meta=meta, frames=df))
    return out


def _null_replicate(args) -> float:
    dev, held, law, em, delay, light, b, seed, obs_pos_sd, obs_heading_sd = args
    st = replace(light, delay=delay)
    sim_dev = replay_dataset(
        dev, law, em, delay, seed + b, light.manoeuvre.vel_window, obs_pos_sd, obs_heading_sd
    )
    sim_held = replay_dataset(
        held, law, em, delay, seed + 1000 + b, light.manoeuvre.vel_window, obs_pos_sd, obs_heading_sd
    )
    fits = {f"single:{k}": fit_single(sim_dev, k, st) for k in LAWS}
    sw = fit_switching(sim_dev, st, transitions="plain")
    start = delay + 2
    ll_sw = score(sim_held, sw, start=start, mp=light.manoeuvre)
    best = max(
        (score(sim_held, f, start=start, mp=light.manoeuvre) for f in fits.values()),
        key=lambda s: s["loglik"],
    )
    return (ll_sw["loglik"] - best["loglik"]) / max(ll_sw["n_frames"], 1)


def switching_null(
    dev: list[Chase],
    held: list[Chase],
    law: str,
    em: Emissions,
    delay: int,
    light: FitSettings,
    B: int,
    seed: int,
    obs_pos_sd: float,
    obs_heading_sd: float,
    verbose: bool = False,
    jobs: int = 1,
) -> np.ndarray:
    """Held-out per-frame gap (switching minus best single law) under the single-law null. The
    replicates are independent and run in ``jobs`` processes when more than one is asked for."""
    tasks = [(dev, held, law, em, delay, light, b, seed, obs_pos_sd, obs_heading_sd) for b in range(B)]
    gaps = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            for i, g in enumerate(ex.map(_null_replicate, tasks)):
                gaps.append(g)
                if verbose and (i + 1) % 20 == 0:
                    print(f"  null replicate {i + 1} of {B}", flush=True)
    else:
        for i, t in enumerate(tasks):
            gaps.append(_null_replicate(t))
            if verbose and (i + 1) % 20 == 0:
                print(f"  null replicate {i + 1} of {B}", flush=True)
    return np.array(gaps)


# ----------------------------------------------------------------------------- posteriors and summaries


def posterior_tables(
    chases: list[Chase], fit: Fit, mp: FeatureParams, v_min: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    post = posteriors(chases, fit, mp)
    roles = state_roles(fit)
    steering_states = np.array([r == "steering" for r in roles])
    by_id = {c.meta.chase_id: c for c in chases}
    frames, summaries = [], []
    for p in post:
        c = by_id[p["chase_id"]]
        s = p["s"]
        g = p["gamma"]
        inf = informative_mask(c, v_min)[s]
        steer = steering_probability(g, fit)
        filt = p["filtered"]
        steer_f = steering_probability(filt, fit)
        arg = np.argmax(g, axis=1)
        path = p["path"]
        fr = c.frames
        cols = {
            "chase_id": p["chase_id"],
            "frame": s,
            "t": fr["t"].to_numpy()[s],
        }
        for k in range(g.shape[1]):
            cols[f"p_{STATE_NAMES[k]}"] = g[:, k]
        cols.update(
            {
                "p_steering": steer,
                "p_steering_filtered": steer_f,
                "path": path,
                "steering_path": steering_states[path].astype(int),
                "steering_filtered": steering_states[np.argmax(filt, axis=1)].astype(int),
                "informative": inf.astype(int),
                "range": fr["range"].to_numpy()[s],
                "t_speed": fr["t_speed"].to_numpy()[s],
                "px": fr["px"].to_numpy()[s],
                "py": fr["py"].to_numpy()[s],
                "tx": fr["tx"].to_numpy()[s],
                "ty": fr["ty"].to_numpy()[s],
                "p_heading": fr["p_heading"].to_numpy()[s],
                "p_speed": fr["p_speed"].to_numpy()[s],
            }
        )
        frames.append(pd.DataFrame(cols))
        summary = {
            "chase_id": p["chase_id"],
            "animal_id": c.meta.animal_id,
            "session_id": c.meta.session_id,
            "n_frames": len(s),
            "n_informative": int(inf.sum()),
            "duration_s": c.duration_s,
            "end_reason": c.meta.end_reason,
            "p_steering_mean": float(steer.mean()),
            "p_steering_informative": float(steer[inf].mean()) if inf.any() else np.nan,
            "frac_steering_path": float(steering_states[path].mean()),
        }
        for k in range(g.shape[1]):
            summary[f"frac_{STATE_NAMES[k]}"] = float((arg == k).mean())
        summary.update(
            {
                "n_switches": int((np.diff(path) != 0).sum()),
                "dominant": STATE_NAMES[int(np.bincount(arg, minlength=g.shape[1]).argmax())],
                "mean_range": float(c.frames["range"].mean()),
                "mean_t_speed": float(c.frames["t_speed"].mean()),
            }
        )
        summaries.append(summary)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def state_regression(chases: list[Chase], fit: Fit, mp: FeatureParams) -> pd.DataFrame:
    """Within each decoded state, the least-squares slope and the correlation of the heading
    increment on the pure-pursuit regressor (the bearing error at the delay): the evidence that a
    state steers toward the target or does not."""
    arrs = arrays(chases, mp)
    data = _designs(arrs, fit.delay, fit.emissions.tau, fit.delay + 2)
    post = {p["chase_id"]: p for p in posteriors(chases, fit, mp)}
    ys, xs, ks = [], [], []
    for a, y, X, _s, _yp, _Xp in data:
        p = post.get(a.chase_id)
        if p is None or len(p["path"]) != len(y):
            continue
        ys.append(y)
        xs.append(X[:, PURE])
        ks.append(np.asarray(p["path"]))
    y, x, k = np.concatenate(ys), np.concatenate(xs), np.concatenate(ks)
    rows = []
    for j, role in enumerate(state_roles(fit)):
        m = k == j
        row = {"state": STATE_NAMES[j], "role": role, "n_frames": int(m.sum()), "share": float(m.mean())}
        if m.sum() >= 3:
            xm, ym = x[m], y[m]
            sxx = float(np.sum((xm - xm.mean()) ** 2))
            row["slope"] = float(np.sum((xm - xm.mean()) * (ym - ym.mean())) / sxx) if sxx > 0 else np.nan
            row["corr"] = float(np.corrcoef(xm, ym)[0, 1]) if sxx > 0 else np.nan
            row["abs_bearing_median_rad"] = float(np.median(np.abs(xm)))
        else:
            row.update({"slope": np.nan, "corr": np.nan, "abs_bearing_median_rad": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def animal_summary(summ: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "n_chases": ("chase_id", "size"),
        "n_frames": ("n_frames", "sum"),
        "n_informative": ("n_informative", "sum"),
        "p_steering_mean": ("p_steering_mean", "mean"),
        "p_steering_informative": ("p_steering_informative", "mean"),
        "frac_steering_path": ("frac_steering_path", "mean"),
    }
    for name in STATE_NAMES:
        if f"frac_{name}" in summ.columns:
            agg[f"frac_{name}"] = (f"frac_{name}", "mean")
    agg["switches_per_chase"] = ("n_switches", "mean")
    return summ.groupby("animal_id").agg(**agg).reset_index()


def event_aligned(
    chases: list[Chase],
    frames_post: pd.DataFrame,
    mp: FeatureParams,
    before: int = 15,
    after: int = 30,
    B: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean steering probability around target-manoeuvre onsets with a chase-level bootstrap band,
    and the curve predicted from chase phase alone: the mean steering probability of all chases
    as a function of the fraction of the chase elapsed (twenty bins), read off at the phase of
    each lag around each onset. The difference between the two is the part of the curve that the
    decline of steering over a chase does not explain."""
    rng = np.random.default_rng(seed)
    lags = np.arange(-before, after + 1)
    per_chase, per_chase_phase = [], []
    post_by = {cid: g.set_index("frame") for cid, g in frames_post.groupby("chase_id")}
    n_bins = 20
    phase_x, phase_y = [], []
    for g in post_by.values():
        f = g.index.to_numpy()
        phase_x.append((f - f.min()) / max(f.max() - f.min(), 1))
        phase_y.append(g["p_steering"].to_numpy())
    px, py = np.concatenate(phase_x), np.concatenate(phase_y)
    pbin = np.clip((px * n_bins).astype(int), 0, n_bins - 1)
    profile = np.array([py[pbin == b].mean() if np.any(pbin == b) else np.nan for b in range(n_bins)])
    for c in chases:
        if c.meta.chase_id not in post_by:
            continue
        arr = prepare(c, mp)
        onsets = np.flatnonzero(arr.manoeuvre)
        if len(onsets) == 0:
            continue
        g = post_by[c.meta.chase_id]
        steer = g["p_steering"]
        steer = steer[~steer.index.duplicated()]
        f0, f1 = int(steer.index.min()), int(steer.index.max())
        curves, phase_curves = [], []
        for m in onsets:
            fr = m + lags
            curves.append(steer.reindex(fr).to_numpy(dtype=float))
            inside = (fr >= f0) & (fr <= f1)
            pc = np.full(len(fr), np.nan)
            pb = np.clip(((fr[inside] - f0) / max(f1 - f0, 1) * n_bins).astype(int), 0, n_bins - 1)
            pc[inside] = profile[pb]
            phase_curves.append(pc)
        per_chase.append(np.nanmean(np.array(curves), axis=0))
        per_chase_phase.append(np.nanmean(np.array(phase_curves), axis=0))
    if not per_chase:
        return pd.DataFrame(
            {
                "lag_frames": lags,
                "mean": np.nan,
                "lo": np.nan,
                "hi": np.nan,
                "phase_only": np.nan,
                "n_chases": 0,
            }
        )
    M = np.array(per_chase)
    M_phase = np.array(per_chase_phase)
    mean = np.nanmean(M, axis=0)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(M), len(M))
        boots.append(np.nanmean(M[idx], axis=0))
    boots = np.array(boots)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return pd.DataFrame(
        {
            "lag_frames": lags,
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "phase_only": np.nanmean(M_phase, axis=0),
            "n_chases": len(M),
            "n_at_lag": np.sum(~np.isnan(M), axis=0),
        }
    )


def params_by_bin(chases: list[Chase], base: Fit, st: FitSettings, B: int, seed: int) -> pd.DataFrame:
    """Lead time and navigation constant refitted within chase-level tertiles of mean target
    speed and mean range, with bootstrap intervals, plus the steering probability in each bin."""
    rows = []
    feats = pd.DataFrame(
        {
            "chase_id": [c.meta.chase_id for c in chases],
            "t_speed": [float(c.frames["t_speed"].mean()) for c in chases],
            "range": [float(c.frames["range"].mean()) for c in chases],
        }
    )
    by_id = {c.meta.chase_id: c for c in chases}
    for var in ("t_speed", "range"):
        q = feats[var].quantile([1 / 3, 2 / 3]).to_numpy()
        labels = np.digitize(feats[var].to_numpy(), q)
        for b in range(3):
            ids = feats.loc[labels == b, "chase_id"].tolist()
            sub = [by_id[i] for i in ids]
            if len(sub) < 10:
                continue
            fit = fit_switching(
                sub, replace(st, n_restarts=1), transitions="plain", init=base.emissions, init_A=base.A
            )
            boots = bootstrap(
                sub,
                lambda cs: fit_switching(
                    cs,
                    replace(st, n_restarts=1, n_iter=25),
                    transitions="plain",
                    init=base.emissions,
                    init_A=base.A,
                ),
                B=B,
                seed=seed + b,
            )
            taus = [f.emissions.tau for f in boots]
            Ns = [f.emissions.N for f in boots]
            gains = [f.emissions.gain for f in boots]
            lo_t, hi_t = interval(taus)
            lo_n, hi_n = interval(Ns)
            lo_g, hi_g = interval(gains)
            post = posteriors(sub, fit, st.manoeuvre)
            p_steer = float(np.mean(np.concatenate([steering_probability(p["gamma"], fit) for p in post])))
            rows.append(
                {
                    "variable": var,
                    "bin": b,
                    "bin_lo": float(feats.loc[labels == b, var].min()),
                    "bin_hi": float(feats.loc[labels == b, var].max()),
                    "n_chases": len(sub),
                    "tau": fit.emissions.tau,
                    "tau_lo": lo_t,
                    "tau_hi": hi_t,
                    "N": fit.emissions.N,
                    "N_lo": lo_n,
                    "N_hi": hi_n,
                    "gain": fit.emissions.gain,
                    "gain_lo": lo_g,
                    "gain_hi": hi_g,
                    "p_steering": p_steer,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the pipeline


def run_dataset(
    dataset: str,
    dev_split: str,
    held_split: str | None,
    out_dir: Path | None = None,
    cfg: dict | None = None,
    frozen_delay: int | None = None,
    verbose: bool = True,
    persistence: bool = False,
    variant: str | None = None,
    null_B: int | None = None,
    bootstrap_B: int | None = None,
    bins_B: int | None = None,
    jobs: int = 1,
    reuse_null: bool = False,
) -> dict:
    """Fit, score, test and summarise one dataset. ``reuse_null`` keeps the parametric-null gaps
    of an earlier run of the same analysis (they depend on the best single law, which is refitted
    on replays with the light settings, not on the switching fit) and recomputes the p-value
    against the new observed gap. ``persistence`` adds the fourth state of the
    post-freeze follow-up; ``variant`` selects an alternative loading of the data (the arena-fit
    scale of the primary dataset); the ``*_B`` arguments override the replicate counts in the
    config."""
    cfg = cfg or load_cfg()
    full, light, delays = settings_from_cfg(cfg)
    rc = cfg.get("real", {})
    mp = full.manoeuvre
    v_min = float(rc.get("informative_speed", INFORMATIVE_SPEED))
    B_param = int(
        bootstrap_B if bootstrap_B is not None else _per_dataset(rc.get("bootstrap_params"), dataset, 100)
    )
    B_null = int(null_B if null_B is not None else _per_dataset(rc.get("null_B"), dataset, 200))
    B_bin = int(bins_B if bins_B is not None else _per_dataset(rc.get("bootstrap_bins"), dataset, 30))
    seed = int(rc.get("seed", 0))
    if out_dir is None:
        base = RESULTS_FOLLOWUP if persistence else RESULTS
        out_dir = base / (dataset if variant is None else f"{dataset}_{variant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    tag = f"{dataset}{'+persistence' if persistence else ''}{'/' + variant if variant else ''}"

    dev = load_real(dataset, dev_split, variant=variant)
    held = load_real(dataset, held_split, variant=variant) if held_split else dev
    in_sample = held_split is None
    report = {
        "dataset": dataset,
        "variant": variant,
        "persistence": persistence,
        "dev_split": dev_split,
        "held_split": held_split or dev_split,
        "in_sample_held": in_sample,
        "dev": summarise_split(dev, v_min),
        "held": summarise_split(held, v_min),
        "informative_speed": v_min,
    }
    full = replace(full, persistence=persistence)
    light = replace(light, persistence=persistence)

    # 1. delay
    if frozen_delay is None:
        log(f"[{tag}] delay profile on {len(dev)} development chases")
        prof = delay_profile(dev, delays, light, model="switching:plain")
        d = best_delay(prof)
        pd.DataFrame(
            [
                {
                    "delay": r["delay"],
                    "loglik_per_frame": r["loglik"] / r["n_frames"],
                    "n_frames": r["n_frames"],
                }
                for r in prof
            ]
        ).to_csv(out_dir / "delay_profile.csv", index=False)
    else:
        d = int(frozen_delay)
    st = replace(full, delay=d)
    report["delay_frames"] = d
    report["delay_ms"] = 1000.0 * d / float(dev[0].meta.fps)
    report["delay_frozen"] = frozen_delay is not None

    # 2. fits on dev
    log(f"[{tag}] fitting models at delay {d}")
    fits, ar = fit_models(dev, st, mp)
    report["fits"] = {k: f.summary() for k, f in fits.items()}
    report["fits"]["switching_ar"] = ar.summary()
    bs = best_single(fits)
    report["best_single_dev"] = bs
    sw = fits["switching:plain"]
    report["state_roles"] = state_roles(sw)

    # 3. bootstrap intervals for the development parameters: every refit starts from the fitted
    #    model with the full settings, so that the interval is built around the point estimate
    log(f"[{tag}] bootstrap of parameters, B = {B_param}")
    boots = bootstrap(
        dev,
        lambda cs: fit_switching(
            cs, replace(st, n_restarts=1), transitions="plain", init=sw.emissions, init_A=sw.A
        ),
        B=B_param,
        seed=seed,
        strata=chase_strata(dev),
    )
    ci = {}
    for name in ("gain", "tau", "N", "sigma"):
        vals = [getattr(f.emissions, name) for f in boots]
        est = float(getattr(sw.emissions, name))
        lo, hi = interval(vals)
        ci[name] = {"estimate": est, "lo": lo, "hi": hi, "contains_estimate": bool(lo <= est <= hi)}
    stay = np.array([np.diag(f.A) for f in boots])
    ci["stay_probability"] = {
        "estimate": np.diag(sw.A).tolist(),
        "lo": np.percentile(stay, 2.5, axis=0).tolist(),
        "hi": np.percentile(stay, 97.5, axis=0).tolist(),
    }
    single_boots = bootstrap(
        dev,
        lambda cs: fit_single(cs, bs.split(":")[1], st, init=fits[bs].emissions),
        B=B_param,
        seed=seed + 1,
        strata=chase_strata(dev),
    )
    ci["best_single"] = {}
    for name in ("gain", "tau", "N", "sigma"):
        est = float(getattr(fits[bs].emissions, name))
        lo, hi = interval([getattr(f.emissions, name) for f in single_boots])
        ci["best_single"][name] = {
            "estimate": est,
            "lo": lo,
            "hi": hi,
            "contains_estimate": bool(lo <= est <= hi),
        }
    report["bootstrap"] = ci

    # 4. scores on held (one pass) and on dev
    log(f"[{tag}] scoring on {len(held)} held-out chases")
    models = [*list(fits), "switching_ar"]
    sc_held = per_chase_scores(held, fits, ar, d, mp)
    sc_dev = per_chase_scores(dev, fits, ar, d, mp)
    sc_held.to_csv(out_dir / "scores_held.csv", index=False)
    comp_held = comparison_table(sc_held, models, seed=seed)
    comp_dev = comparison_table(sc_dev, models, seed=seed)
    comp_held.to_csv(out_dir / "comparison_held.csv", index=False)
    comp_dev.to_csv(out_dir / "comparison_dev.csv", index=False)
    report["comparison_held"] = comp_held.to_dict(orient="records")
    sc_inf = masked_scores(held, fits, ar, d, mp, v_min)
    if len(sc_inf) >= 5:
        comp_inf = comparison_table(sc_inf, models, seed=seed)
        comp_inf.to_csv(out_dir / "comparison_held_informative.csv", index=False)
        report["comparison_held_informative"] = comp_inf.to_dict(orient="records")

    # 5. switching null by parametric bootstrap along the recorded target paths
    gap_obs = float(comp_held.set_index("model").loc["switching:plain", "diff_vs_best_single_per_frame"])
    law_null = bs.split(":")[1]
    previous = None
    if reuse_null and (out_dir / "fit_report.json").exists():
        with open(out_dir / "fit_report.json", encoding="utf-8") as f:
            previous = json.load(f).get("switching_null")
        if previous is not None and (
            previous.get("law") != law_null or len(previous.get("null_gaps", [])) == 0
        ):
            previous = None
    if previous is not None:
        log(f"[{tag}] reusing the parametric null of the previous run, B = {len(previous['null_gaps'])}")
        gaps = np.asarray(previous["null_gaps"], dtype=float)
    else:
        log(f"[{tag}] parametric null under {law_null}, B = {B_null}")
        gaps = switching_null(
            dev,
            held,
            law_null,
            fits[bs].emissions,
            d,
            light,
            B_null,
            seed,
            float(rc.get("obs_pos_sd", 0.2)),
            float(rc.get("obs_heading_sd", 0.05)),
            verbose=verbose,
            jobs=jobs,
        )
    report["switching_null"] = {
        "law": law_null,
        "gap_observed_per_frame": gap_obs,
        "null_gaps": gaps.tolist(),
        "null_q95": float(np.quantile(gaps, 0.95)),
        "null_mean": float(gaps.mean()),
        "null_sd": float(gaps.std()),
        "p_value": float((np.sum(gaps >= gap_obs) + 1) / (len(gaps) + 1)),
        "B": len(gaps),
        "obs_pos_sd": float(rc.get("obs_pos_sd", 0.2)),
        "obs_heading_sd": float(rc.get("obs_heading_sd", 0.05)),
    }

    # 6. posteriors on held from the development parameters
    log(f"[{tag}] posteriors")
    frames_post, summ = posterior_tables(held, sw, mp, v_min)
    _frames_in, summ_in = posterior_tables(held, fits["switching:input"], mp, v_min)
    summ["p_steering_mean_input"] = summ_in["p_steering_mean"].to_numpy()
    summ.to_csv(out_dir / "posteriors_chase.csv", index=False)
    animal_summary(summ).to_csv(out_dir / "posteriors_animal.csv", index=False)
    derived = DERIVED_DIR / dataset
    derived.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{held_split or dev_split}{'_persistence' if persistence else ''}{'_' + variant if variant else ''}"
    )
    frames_post.to_parquet(derived / f"posteriors_frames_{suffix}.parquet", index=False)
    keep = (
        summ.sort_values("n_frames", ascending=False)["chase_id"]
        .head(int(rc.get("frames_csv_chases", 300)))
        .tolist()
    )
    frames_post[frames_post["chase_id"].isin(keep)].round(4).to_csv(
        out_dir / "posteriors_frames.csv", index=False
    )
    post_summary = {
        "state_roles": state_roles(sw),
        "p_steering_mean": float(summ["p_steering_mean"].mean()),
        "p_steering_informative": float(np.nanmean(summ["p_steering_informative"])),
        "frac_steering_path": float(summ["frac_steering_path"].mean()),
        "switches_per_chase": float(summ["n_switches"].mean()),
        "chases_with_switch": float((summ["n_switches"] > 0).mean()),
        "dominant_counts": summ["dominant"].value_counts().to_dict(),
    }
    for name in STATE_NAMES:
        if f"frac_{name}" in summ.columns:
            post_summary[f"frac_{name}"] = float(summ[f"frac_{name}"].mean())
    report["posteriors"] = post_summary
    sr = state_regression(held, sw, mp)
    sr.to_csv(out_dir / "state_regression.csv", index=False)
    report["state_regression"] = sr.to_dict(orient="records")

    # 7. steering probability around target manoeuvres
    ev = event_aligned(held, frames_post, mp, seed=seed)
    ev.to_csv(out_dir / "event_aligned.csv", index=False)
    report["event_aligned_n_chases"] = int(ev["n_chases"].iloc[0]) if len(ev) else 0

    # 8. parameters by bins of target speed and range (development chases)
    log(f"[{tag}] parameters by bin, B = {B_bin}")
    pb = params_by_bin(dev, sw, st, B_bin, seed)
    pb.to_csv(out_dir / "params_by_bin.csv", index=False)

    with open(out_dir / "fit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=float)
    log(f"[{tag}] done")
    return report


# ----------------------------------------------------------------------------- leave-one-animal-out


def leave_one_animal_out(
    dataset: str,
    split: str,
    delay: int,
    cfg: dict | None = None,
    out_dir: Path | None = None,
    persistence: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sensitivity for a replication set scored in sample: every model is refitted on all animals
    but one, at the frozen delay, and scored on the left-out animal; the per-chase scores are
    pooled over the folds and compared against the best single law with chase-level bootstrap
    intervals, as in the main comparison table."""
    cfg = cfg or load_cfg()
    full, _light, _delays = settings_from_cfg(cfg)
    rc = cfg.get("real", {})
    mp = full.manoeuvre
    v_min = float(rc.get("informative_speed", INFORMATIVE_SPEED))
    seed = int(rc.get("seed", 0))
    out_dir = out_dir or ((RESULTS_FOLLOWUP if persistence else RESULTS) / dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    chases = load_real(dataset, split)
    st = replace(full, delay=int(delay), persistence=persistence)
    animals = sorted({c.meta.animal_id for c in chases})
    held_scores, held_inf, folds = [], [], []
    for a in animals:
        train = [c for c in chases if c.meta.animal_id != a]
        test = [c for c in chases if c.meta.animal_id == a]
        log(f"[{dataset}] leave-one-animal-out: holding out {a} ({len(test)} chases)")
        fits, ar = fit_models(train, st, mp)
        sc = per_chase_scores(test, fits, ar, int(delay), mp)
        sc["fold"] = a
        held_scores.append(sc)
        si = masked_scores(test, fits, ar, int(delay), mp, v_min)
        si["fold"] = a
        held_inf.append(si)
        folds.append(
            {
                "held_out_animal": a,
                "n_train_chases": len(train),
                "n_test_chases": len(test),
                "best_single_train": best_single(fits),
                "gain": fits["switching:plain"].emissions.gain,
                "tau": fits["switching:plain"].emissions.tau,
                "N": fits["switching:plain"].emissions.N,
            }
        )
    models = [
        "null",
        *[f"single:{law}" for law in LAWS],
        "switching:plain",
        "switching:input",
        "switching_ar",
    ]
    pooled = comparison_table(pd.concat(held_scores, ignore_index=True), models, seed=seed)
    pooled["frames"] = "all"
    inf = pd.concat(held_inf, ignore_index=True)
    rows = [pooled]
    if len(inf) >= 5:
        pin = comparison_table(inf, models, seed=seed)
        pin["frames"] = "informative"
        rows.append(pin)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out_dir / "leave_one_animal_out.csv", index=False)
    pd.DataFrame(folds).to_csv(out_dir / "leave_one_animal_out_folds.csv", index=False)
    return table


# ----------------------------------------------------------------------------- refreshing the posteriors


def fit_from_summary(d: dict) -> Fit:
    """Rebuild a Fit from the summary stored in fit_report.json (parameters, transitions, delay)."""
    em = Emissions(**{k: float(d[k]) for k in ("gain", "tau", "N", "sigma", "rho", "eps", "sigma_out")})
    return Fit(
        d["model"],
        em,
        int(d["delay"]),
        float(d["loglik"]),
        int(d["n_frames"]),
        pi=np.asarray(d["pi"]) if "pi" in d else None,
        A=np.asarray(d["A"]) if "A" in d else None,
        W=np.asarray(d["W"]) if "W" in d else None,
        law=d["model"].split(":")[1] if d["model"].startswith("single") else None,
        persistence=bool(d.get("persistence", False)),
    )


def refresh_posteriors(
    dataset: str,
    dev_split: str,
    held_split: str | None,
    out_dir: Path | None = None,
    cfg: dict | None = None,
    persistence: bool = False,
    variant: str | None = None,
    verbose: bool = True,
) -> dict:
    """Recompute the posterior tables, the per-animal summary, the per-frame parquet and csv and
    the event-aligned curve from the parameters already in fit_report.json, without refitting."""
    cfg = cfg or load_cfg()
    full, _light, _delays = settings_from_cfg(cfg)
    rc = cfg.get("real", {})
    mp = full.manoeuvre
    v_min = float(rc.get("informative_speed", INFORMATIVE_SPEED))
    seed = int(rc.get("seed", 0))
    if out_dir is None:
        base = RESULTS_FOLLOWUP if persistence else RESULTS
        out_dir = base / (dataset if variant is None else f"{dataset}_{variant}")
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    with open(out_dir / "fit_report.json", encoding="utf-8") as f:
        report = json.load(f)
    held = load_real(dataset, held_split or dev_split, variant=variant)
    sw = fit_from_summary(report["fits"]["switching:plain"])
    sw_in = fit_from_summary(report["fits"]["switching:input"])
    log(f"[{dataset}] refreshing posteriors on {len(held)} chases")
    frames_post, summ = posterior_tables(held, sw, mp, v_min)
    _frames_in, summ_in = posterior_tables(held, sw_in, mp, v_min)
    summ["p_steering_mean_input"] = summ_in["p_steering_mean"].to_numpy()
    summ.to_csv(out_dir / "posteriors_chase.csv", index=False)
    animal_summary(summ).to_csv(out_dir / "posteriors_animal.csv", index=False)
    derived = DERIVED_DIR / dataset
    derived.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{held_split or dev_split}{'_persistence' if persistence else ''}{'_' + variant if variant else ''}"
    )
    frames_post.to_parquet(derived / f"posteriors_frames_{suffix}.parquet", index=False)
    keep = (
        summ.sort_values("n_frames", ascending=False)["chase_id"]
        .head(int(rc.get("frames_csv_chases", 300)))
        .tolist()
    )
    frames_post[frames_post["chase_id"].isin(keep)].round(4).to_csv(
        out_dir / "posteriors_frames.csv", index=False
    )
    post_summary = {
        "state_roles": state_roles(sw),
        "p_steering_mean": float(summ["p_steering_mean"].mean()),
        "p_steering_informative": float(np.nanmean(summ["p_steering_informative"])),
        "frac_steering_path": float(summ["frac_steering_path"].mean()),
        "switches_per_chase": float(summ["n_switches"].mean()),
        "chases_with_switch": float((summ["n_switches"] > 0).mean()),
        "dominant_counts": summ["dominant"].value_counts().to_dict(),
    }
    for name in STATE_NAMES:
        if f"frac_{name}" in summ.columns:
            post_summary[f"frac_{name}"] = float(summ[f"frac_{name}"].mean())
    report["posteriors"] = post_summary
    sr = state_regression(held, sw, mp)
    sr.to_csv(out_dir / "state_regression.csv", index=False)
    report["state_regression"] = sr.to_dict(orient="records")
    ev = event_aligned(held, frames_post, mp, seed=seed)
    ev.to_csv(out_dir / "event_aligned.csv", index=False)
    report["event_aligned_n_chases"] = int(ev["n_chases"].iloc[0]) if len(ev) else 0
    with open(out_dir / "fit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=float)
    log(f"[{dataset}] posteriors refreshed")
    return report


def keep_report_only(out_dir: Path) -> None:
    """Remove every output of a run except its fit report (and a licence notice): the follow-up
    and the sensitivity runs are summarised by their reports, and no figure or table reads their
    per-chase or per-frame files."""
    for f in out_dir.iterdir():
        if f.is_file() and f.name not in ("fit_report.json", "LICENSE"):
            f.unlink()
