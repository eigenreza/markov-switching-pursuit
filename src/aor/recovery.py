"""The recovery study on the synthetic suite.

For every condition the replicates are split in two: even replicates fit the models, odd
replicates are held out. The study reports

(a) law classification of held-out chases by the single-law likelihoods and by the switching
    posterior, across noise levels and truncation lengths;
(b) switching detection: the per-frame held-out log-likelihood gap between the switching model
    and the best single law, with the detection threshold set at a high quantile of that gap
    on single-law data (the null), the detection rate when switching is present, the false
    detection rate on misspecified targets, and the switch-time error;
(c) parameter recovery for the delay, the lead time and the navigation constant, with
    chase-level bootstrap intervals and their coverage;
(d) posterior calibration of the law probabilities;
plus the recovery limits: the shortest length and the highest noise at which the targets hold.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .contracts import Chase
from .inference.features import FeatureParams
from .inference.models import (
    FitSettings,
    best_delay,
    bootstrap,
    delay_profile,
    fit_null,
    fit_single,
    fit_switching,
    interval,
    posteriors,
    score,
)
from .laws import LAW_INDEX, LAWS
from .synthetic import OUT_DIR as SUITE_DIR
from .synthetic import chases_from_suite, load_suite

ROOT = Path(__file__).resolve().parents[2]
LAW_PARAMS = {"pure_pursuit": 1, "lead_pursuit": 2, "proportional_navigation": 1}
CONFIG = ROOT / "configs" / "model.yaml"
OUT_DIR = ROOT / "results" / "recovery"


def load_settings(config_path: Path = CONFIG) -> tuple[FitSettings, FitSettings, dict]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
    return full, light, {**cfg["recovery"], "delays": list(inf["delays"])}


def truncate(chase: Chase, n: int) -> Chase:
    df = chase.frames.iloc[:n].reset_index(drop=True)
    meta = replace(
        chase.meta,
        end_frame=int(df["frame"].iloc[-1]),
        end_reason="clip_end",
        chase_id=f"{chase.meta.chase_id}@{n}",
    )
    return Chase(meta=meta, frames=df)


def fit_condition(
    train: list[Chase], full: FitSettings, light: FitSettings, delays: list[int], with_input: bool
) -> dict:
    """Profile the delay with the switching model, then fit every model at the best delay."""
    prof = delay_profile(train, delays, light, model="switching:plain")
    d = best_delay(prof)
    st = replace(full, delay=d)
    fits = {"null": fit_null(train, st)}
    for law in LAWS:
        fits[f"single:{law}"] = fit_single(train, law, st)
    fits["switching:plain"] = fit_switching(train, st, transitions="plain")
    if with_input:
        sw = fits["switching:plain"]
        fits["switching:input"] = fit_switching(
            train, replace(st, n_restarts=1), transitions="input", init=sw.emissions, init_A=sw.A
        )
    return {"delay": d, "profile": [(r["delay"], r["loglik"], r["n_frames"]) for r in prof], "fits": fits}


def evaluate(test: list[Chase], fits: dict, delay: int) -> pd.DataFrame:
    """Per held-out chase: log-likelihood under each model on the common support, the predicted
    law, and the switching gap per frame."""
    start = delay + 2
    scores = {name: score(test, f, start=start) for name, f in fits.items()}
    rows = []
    for c in test:
        cid = c.meta.chase_id
        if cid not in scores["null"]["per_chase"]:
            continue
        n = scores["null"]["per_chase"][cid][1]
        single = {law: scores[f"single:{law}"]["per_chase"][cid][0] for law in LAWS}
        # lead pursuit nests pure pursuit (tau = 0), so each chase is classified by a BIC-penalised
        # score: log-likelihood minus half the law's own parameter count times log(frames)
        penalised = {law: single[law] - 0.5 * LAW_PARAMS[law] * np.log(n) for law in LAWS}
        best_law = max(penalised, key=penalised.get)
        sw = scores["switching:plain"]["per_chase"][cid][0]
        rows.append(
            {
                "chase_id": cid,
                "n_frames": n,
                "pred_law": best_law,
                "ll_best_single": single[best_law],
                "ll_switching": sw,
                "ll_null": scores["null"]["per_chase"][cid][0],
                "gap_per_frame": (sw - single[best_law]) / n,
                **{f"ll_{law}": v for law, v in single.items()},
            }
        )
    return pd.DataFrame(rows)


def posterior_accuracy(test: list[Chase], fit) -> tuple[float, list[dict]]:
    post = posteriors(test, fit)
    correct, total = 0, 0
    for p in post:
        if p["law_true"] is None:
            continue
        pred = np.argmax(p["gamma"], axis=1)
        correct += int(np.sum(pred == p["law_true"].astype(int)))
        total += len(pred)
    return (correct / total if total else float("nan")), post


def first_switch_frame(path: np.ndarray, s: np.ndarray, from_law: int, to_law: int) -> int | None:
    for i in range(1, len(path)):
        if path[i - 1] == from_law and path[i] == to_law:
            return int(s[i])
    return None


def run(
    out_dir: Path = OUT_DIR,
    suite_dir: Path = SUITE_DIR,
    config_path: Path = CONFIG,
    max_conditions: int | None = None,
    verbose: bool = True,
) -> dict:
    full, light, rc = load_settings(config_path)
    delays = rc["delays"]
    frames, meta = load_suite(suite_dir)
    chases = chases_from_suite(frames, meta)
    by_id = {c.meta.chase_id: c for c in chases}
    conditions = list(dict.fromkeys(meta["condition"]))
    if max_conditions:
        conditions = conditions[:max_conditions]
    rows_cls, rows_det, rows_sw, rows_par, rows_prof, rows_cal, rows_w = [], [], [], [], [], [], []
    examples = {}
    calib_pred, calib_true = [], []

    for ci, cond in enumerate(conditions):
        m = meta[meta["condition"] == cond]
        group = str(m["animal_id"].iloc[0])
        law = str(m["law"].iloc[0])
        noise = float(m["noise"].iloc[0])
        true_delay = int(m["delay"].iloc[0])
        train = [by_id[i] for i in m[m["replicate"] % 2 == 0]["chase_id"]]
        test = [by_id[i] for i in m[m["replicate"] % 2 == 1]["chase_id"]]
        if len(train) < 3 or len(test) < 3:
            continue
        prefix = cond.split("__")[0]
        with_input = prefix in ("switch_manoeuvre", "switch_manoeuvre_turn", "switch_range", "misspec_jumper")
        if verbose:
            print(f"[{ci + 1}/{len(conditions)}] {cond}: train {len(train)} test {len(test)}", flush=True)
        res = fit_condition(train, full, light, delays, with_input)
        d, fits = res["delay"], res["fits"]
        for dd, ll, n in res["profile"]:
            rows_prof.append(
                {
                    "condition": cond,
                    "group": group,
                    "law": law,
                    "noise": noise,
                    "true_delay": true_delay,
                    "delay": dd,
                    "loglik_per_frame": ll / n,
                }
            )
        sw = fits["switching:plain"]

        # (a) classification across truncation lengths (single group); full length for the others
        lengths = [None] + (list(rc["lengths_frames"]) if group in ("single", "realistic") else [])
        for L in lengths:
            subset = [truncate(c, L) for c in test if L is None or len(c.frames) >= L]
            if len(subset) < 3:
                continue
            ev = evaluate(subset, fits, d)
            acc_post, post = posterior_accuracy(subset, sw)
            for _, r in ev.iterrows():
                rows_cls.append(
                    {
                        "condition": cond,
                        "group": group,
                        "noise": noise,
                        "length_frames": L if L else int(r["n_frames"]) + d + 1,
                        "length_tag": L if L else 0,
                        "true_law": law,
                        "pred_law": r["pred_law"],
                        "gap_per_frame": r["gap_per_frame"],
                        "chase_id": r["chase_id"],
                        "n_frames": int(r["n_frames"]),
                    }
                )
            rows_det.append(
                {
                    "condition": cond,
                    "group": group,
                    "prefix": prefix,
                    "law": law,
                    "noise": noise,
                    "length_tag": L if L else 0,
                    "n_test": len(ev),
                    "gap_per_frame_mean": float(ev["gap_per_frame"].mean()),
                    "gap_per_frame_q50": float(ev["gap_per_frame"].median()),
                    "posterior_accuracy": acc_post,
                    "delay_est": d,
                    "true_delay": true_delay,
                    "ll_switching_per_frame": float(ev["ll_switching"].sum() / ev["n_frames"].sum()),
                    "ll_best_single_per_frame": float(ev["ll_best_single"].sum() / ev["n_frames"].sum()),
                    "ll_null_per_frame": float(ev["ll_null"].sum() / ev["n_frames"].sum()),
                    "gain": sw.emissions.gain,
                    "tau": sw.emissions.tau,
                    "N": sw.emissions.N,
                    "sigma": sw.emissions.sigma,
                    "A_stay_mean": float(np.mean(np.diag(sw.A))),
                }
            )
            if L is None and group == "switching":
                for p in post:
                    if p["law_true"] is not None:
                        calib_pred.append(p["gamma"])
                        calib_true.append(p["law_true"].astype(int))
        # (b) switch-time error and covariate recovery
        if prefix == "switch_fixed":
            post = posteriors(test, sw)
            switch_frame = 45
            for p in post:
                f = first_switch_frame(
                    p["path"], p["s"], LAW_INDEX["lead_pursuit"], LAW_INDEX["pure_pursuit"]
                )
                rows_sw.append(
                    {
                        "condition": cond,
                        "noise": noise,
                        "chase_id": p["chase_id"],
                        "true_switch_frame": switch_frame,
                        "decoded_switch_frame": f,
                        "error_frames": None if f is None else f - switch_frame,
                    }
                )
        if with_input and "switching:input" in fits:
            W = fits["switching:input"].W
            rows_w.append(
                {
                    "condition": cond,
                    "prefix": prefix,
                    "noise": noise,
                    "w_lead_to_pure_recent": float(
                        W[LAW_INDEX["lead_pursuit"], LAW_INDEX["pure_pursuit"], 3]
                    ),
                    "w_pn_to_pure_range": float(
                        W[LAW_INDEX["proportional_navigation"], LAW_INDEX["pure_pursuit"], 1]
                    ),
                    "ll_input": fits["switching:input"].loglik,
                    "ll_plain": sw.loglik,
                    "n_frames": sw.n_frames,
                }
            )
        # (c) parameter recovery with bootstrap intervals
        if group == "parameters":
            allc = train + test
            prof = delay_profile(allc, delays, light, model=f"single:{law}")
            d_hat = best_delay(prof)
            st = replace(full, delay=d_hat)
            point = fit_single(allc, law, st)
            B = int(rc["bootstrap_B"])
            fits_b = bootstrap(
                allc, lambda cs, st=st, law=law: fit_single(cs, law, st), B=B, seed=int(rc["seed"]) + ci
            )
            true = {
                "gain": float(m["gain"].iloc[0]),
                "tau": float(m["tau"].iloc[0]),
                "N": float(m["N"].iloc[0]),
            }
            for name in ("gain", "tau", "N"):
                if (
                    (name == "tau" and law != "lead_pursuit")
                    or (name == "N" and law != "proportional_navigation")
                    or (name == "gain" and law == "proportional_navigation")
                ):
                    continue
                vals = np.array([getattr(f.emissions, name) for f in fits_b])
                lo, hi = interval(vals)
                rows_par.append(
                    {
                        "condition": cond,
                        "prefix": prefix,
                        "law": law,
                        "noise": noise,
                        "parameter": name,
                        "true": true[name],
                        "estimate": float(getattr(point.emissions, name)),
                        "lo": lo,
                        "hi": hi,
                        "covered": bool(lo <= true[name] <= hi),
                        "B": B,
                    }
                )
            d_boot = []
            if prefix == "delay":
                for b in range(min(20, B)):
                    rng = np.random.default_rng(1000 + ci * 100 + b)
                    idx = rng.choice(len(allc), size=len(allc), replace=True)
                    pb = delay_profile([allc[i] for i in idx], delays, light, model=f"single:{law}")
                    d_boot.append(best_delay(pb))
                lo, hi = (min(d_boot), max(d_boot)) if d_boot else (d_hat, d_hat)
                rows_par.append(
                    {
                        "condition": cond,
                        "prefix": prefix,
                        "law": law,
                        "noise": noise,
                        "parameter": "delay",
                        "true": true_delay,
                        "estimate": d_hat,
                        "lo": lo,
                        "hi": hi,
                        "covered": bool(lo <= true_delay <= hi),
                        "B": len(d_boot),
                    }
                )

    # pooled fits: one switching model per noise level on the three single-law conditions together,
    # so that the lead time and the navigation constant are identified before posteriors are judged
    rows_pooled = []
    single_meta = meta[meta["animal_id"] == "single"]
    for noise, mn in single_meta.groupby("noise"):
        conds_here = [c for c in conditions if c in set(mn["condition"])]
        if len(set(mn[mn["condition"].isin(conds_here)]["law"])) < 2:
            continue
        train = [
            by_id[i] for i in mn[(mn["replicate"] % 2 == 0) & mn["condition"].isin(conds_here)]["chase_id"]
        ]
        test = [
            by_id[i] for i in mn[(mn["replicate"] % 2 == 1) & mn["condition"].isin(conds_here)]["chase_id"]
        ]
        if verbose:
            print(f"[pooled] noise {noise}: train {len(train)} test {len(test)}", flush=True)
        prof = delay_profile(train, delays, light, model="switching:plain")
        d = best_delay(prof)
        sw = fit_switching(train, replace(full, delay=d), transitions="plain")
        post = posteriors(test, sw)
        truth = {
            c.meta.chase_id: str(meta.loc[meta["chase_id"] == c.meta.chase_id, "law"].iloc[0]) for c in test
        }
        for c, p in zip(test, post, strict=True):
            pred = np.argmax(p["gamma"], axis=1)
            acc = float(np.mean(pred == p["law_true"].astype(int)))
            rows_pooled.append(
                {
                    "noise": float(noise),
                    "law": truth[c.meta.chase_id],
                    "chase_id": c.meta.chase_id,
                    "posterior_accuracy": acc,
                    "dichotomy_accuracy": float(  # anticipatory laws (lead, proportional navigation) against pure
                        np.mean(
                            ((p["gamma"][:, 1] + p["gamma"][:, 2]) > 0.5) == (p["law_true"].astype(int) != 0)
                        )
                    ),
                    "delay_est": d,
                    "tau": sw.emissions.tau,
                    "N": sw.emissions.N,
                    "gain": sw.emissions.gain,
                    "sigma": sw.emissions.sigma,
                }
            )
            calib_pred.append(p["gamma"])
            calib_true.append(p["law_true"].astype(int))
            law = truth[c.meta.chase_id]
            if float(noise) == float(rc["example_noise"]) and law not in examples and len(c.frames) >= 90:
                examples[law] = {
                    "chase_id": c.meta.chase_id,
                    "law": law,
                    "noise": float(noise),
                    "delay": d,
                    "t": c.frames["t"].round(4).tolist(),
                    "px": c.frames["px"].round(3).tolist(),
                    "py": c.frames["py"].round(3).tolist(),
                    "tx": c.frames["tx"].round(3).tolist(),
                    "ty": c.frames["ty"].round(3).tolist(),
                    "s": p["s"].tolist(),
                    "gamma": np.round(p["gamma"], 4).tolist(),
                    "law_true": p["law_true"].astype(int).tolist(),
                }
    pooled = pd.DataFrame(rows_pooled)

    out_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out_dir / "pooled_posteriors.csv", index=False)
    cls = pd.DataFrame(rows_cls)
    det = pd.DataFrame(rows_det)
    cls.to_csv(out_dir / "classification.csv", index=False)
    det.to_csv(out_dir / "detection.csv", index=False)
    pd.DataFrame(rows_sw).to_csv(out_dir / "switch_time.csv", index=False)
    pd.DataFrame(rows_par).to_csv(out_dir / "params.csv", index=False)
    pd.DataFrame(rows_prof).to_csv(out_dir / "delay_profiles.csv", index=False)
    pd.DataFrame(rows_w).to_csv(out_dir / "input_weights.csv", index=False)
    with open(out_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f)

    # (d) calibration
    nb = int(rc["calibration_bins"])
    if calib_pred:
        G = np.concatenate(calib_pred)
        T = np.concatenate(calib_true)
        edges = np.linspace(0, 1, nb + 1)
        for k, law in enumerate(LAWS):
            p = G[:, k]
            hit = (k == T).astype(float)
            which = np.clip(np.digitize(p, edges) - 1, 0, nb - 1)
            for b in range(nb):
                sel = which == b
                if sel.sum():
                    rows_cal.append(
                        {
                            "law": law,
                            "bin_lo": edges[b],
                            "bin_hi": edges[b + 1],
                            "mean_predicted": float(p[sel].mean()),
                            "empirical": float(hit[sel].mean()),
                            "n": int(sel.sum()),
                        }
                    )
    cal = pd.DataFrame(rows_cal)
    cal.to_csv(out_dir / "calibration.csv", index=False)

    # summaries and limits
    summary = summarise(cls, det, pd.DataFrame(rows_sw), pd.DataFrame(rows_par), cal, rc)
    if len(pooled):
        summary["pooled_posterior_accuracy"] = (
            pooled.groupby(["noise", "law"])
            .agg(
                posterior_accuracy=("posterior_accuracy", "mean"),
                dichotomy_accuracy=("dichotomy_accuracy", "mean"),
                n=("chase_id", "size"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        summary["pooled_parameters"] = (
            pooled.groupby("noise")[["delay_est", "tau", "N", "gain", "sigma"]]
            .first()
            .reset_index()
            .to_dict(orient="records")
        )
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, default=float)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    write_limits_md(summary, out_dir / "limits.md")
    if (meta["animal_id"] == "realistic").any():
        realistic_summary(suite_dir, out_dir, config_path, verbose)
        with open(out_dir / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)
    return summary


def summarise(
    cls: pd.DataFrame, det: pd.DataFrame, sw: pd.DataFrame, par: pd.DataFrame, cal: pd.DataFrame, rc: dict
) -> dict:
    out = {"settings": {k: v for k, v in rc.items()}}
    single = cls[cls["group"] == "single"]
    acc = (
        single.assign(correct=lambda d: d["true_law"] == d["pred_law"])
        .groupby(["noise", "length_tag"])
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    out["classification_accuracy"] = acc.to_dict(orient="records")
    conf = (
        single[single["length_tag"] == 0]
        .groupby(["noise", "true_law", "pred_law"])
        .size()
        .reset_index(name="count")
    )
    out["confusion_full_length"] = conf.to_dict(orient="records")
    # detection thresholds per noise level: from clean single-law chases alone, and from the
    # configured null (single-law chases with and without tracking noise)
    level = float(rc["detection_level"])
    full_len = cls[cls["length_tag"] == 0].assign(prefix=lambda d: d["condition"].str.split("__").str[0])
    null_prefixes = list(rc.get("null_prefixes", ["single"]))
    thr_single = (
        full_len[full_len["prefix"] == "single"].groupby("noise")["gap_per_frame"].quantile(level).to_dict()
    )
    thr_null = (
        full_len[full_len["prefix"].isin(null_prefixes)]
        .groupby("noise")["gap_per_frame"]
        .quantile(level)
        .to_dict()
    )
    out["detection_threshold_per_noise"] = {str(k): float(v) for k, v in thr_null.items()}
    out["detection_threshold_single_only_per_noise"] = {str(k): float(v) for k, v in thr_single.items()}
    out["null_prefixes"] = null_prefixes
    det_rows = []
    for (grp, noise), sub in full_len.groupby(["group", "noise"]):
        if noise not in thr_null:
            continue
        det_rows.append(
            {
                "group": grp,
                "noise": noise,
                "detected_fraction": float((sub["gap_per_frame"] > thr_null[noise]).mean()),
                "detected_fraction_single_null": float(
                    (sub["gap_per_frame"] > thr_single.get(noise, np.inf)).mean()
                ),
                "n": len(sub),
            }
        )
    out["detection_by_group"] = det_rows
    by_prefix = []
    for (prefix, noise), sub in full_len[full_len["group"].isin(["switching", "misspecified"])].groupby(
        ["prefix", "noise"]
    ):
        if noise in thr_null:
            by_prefix.append(
                {
                    "prefix": prefix,
                    "noise": noise,
                    "detected_fraction": float((sub["gap_per_frame"] > thr_null[noise]).mean()),
                    "detected_fraction_single_null": float(
                        (sub["gap_per_frame"] > thr_single.get(noise, np.inf)).mean()
                    ),
                    "n": len(sub),
                }
            )
    out["detection_by_condition"] = by_prefix
    out["posterior_accuracy"] = (
        det.groupby(["group", "noise", "length_tag"])["posterior_accuracy"]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    if len(sw):
        err = sw["error_frames"].dropna()
        out["switch_time"] = {
            "n_decoded": int(err.size),
            "n_missing": int(sw["error_frames"].isna().sum()),
            "median_abs_error_frames": float(err.abs().median()) if err.size else None,
            "within_5_frames": float((err.abs() <= 5).mean()) if err.size else None,
            "median_error_frames": float(err.median()) if err.size else None,
        }
    if len(par):
        out["parameter_recovery"] = (
            par.groupby(["parameter"])
            .agg(
                coverage=("covered", "mean"),
                n=("covered", "size"),
                mean_abs_error=("estimate", lambda s: float(np.mean(np.abs(s - par.loc[s.index, "true"])))),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        dl = par[par["parameter"] == "delay"]
        if len(dl):
            out["delay_recovery"] = {
                "exact": float((dl["estimate"] == dl["true"]).mean()),
                "within_1": float(((dl["estimate"] - dl["true"]).abs() <= 1).mean()),
                "n": len(dl),
            }
    if len(cal):
        out["calibration_max_gap"] = float((cal["mean_predicted"] - cal["empirical"]).abs().max())
        out["calibration_weighted_gap"] = float(
            np.average((cal["mean_predicted"] - cal["empirical"]).abs(), weights=cal["n"])
        )
    # limits
    target = float(rc["accuracy_target"])
    limits = {
        "accuracy_target": target,
        "detection_target": float(rc["detection_target"]),
        "false_detection_max": float(rc["false_detection_max"]),
    }
    shortest = {}
    for noise, sub in acc[acc["length_tag"] > 0].groupby("noise"):
        ok = sub[sub["accuracy"] >= target].sort_values("length_tag")
        shortest[str(noise)] = int(ok["length_tag"].iloc[0]) if len(ok) else None
    limits["shortest_length_frames_by_noise"] = shortest
    highest = {}
    for L, sub in acc[acc["length_tag"] > 0].groupby("length_tag"):
        ok = sub[sub["accuracy"] >= target].sort_values("noise")
        highest[str(int(L))] = float(ok["noise"].iloc[-1]) if len(ok) else None
    limits["highest_noise_by_length_frames"] = highest
    dsw = pd.DataFrame(by_prefix)
    if len(dsw):
        fp = dsw[dsw["prefix"].str.startswith("misspec") & ~dsw["prefix"].isin(null_prefixes)]
        limits["false_detection_misspecified_by_noise"] = {
            str(k): float(v) for k, v in fp.groupby("noise")["detected_fraction"].mean().items()
        }
        sw_rows = dsw[dsw["prefix"].str.startswith("switch")]
        limits["detection_by_switch_kind"] = {
            f"{r['prefix']}@{r['noise']}": float(r["detected_fraction"]) for _, r in sw_rows.iterrows()
        }
        limits["switch_kinds_detected_at_target"] = sorted(
            f"{r['prefix']}@{r['noise']}"
            for _, r in sw_rows.iterrows()
            if r["detected_fraction"] >= float(rc["detection_target"])
        )
    out["limits"] = limits
    return out


def _num(v, spec: str) -> str:
    """Format a number for the limits note; a missing value prints as n/a."""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return format(v, spec)
    except (TypeError, ValueError):
        return "n/a"


def write_limits_md(summary: dict, path: Path) -> None:
    """Write the recovery limits as a short results note, every number taken from the summary."""
    lim = summary["limits"]
    acc = pd.DataFrame(summary["classification_accuracy"])
    full = acc[acc["length_tag"] == 0].set_index("noise")["accuracy"]
    pooled = pd.DataFrame(summary.get("pooled_posterior_accuracy", []))
    par = {r["parameter"]: r for r in summary.get("parameter_recovery", [])}
    sw = summary.get("switch_time", {}) or {}
    lines = [
        "# Recovery limits",
        "",
        "Noise is the standard deviation of the heading-increment noise in radians per frame; length is in frames at 30 per second.",
        "",
        "## Law classification of held-out single-law chases",
        "",
    ]
    lines.append(
        "| noise | full length | "
        + " | ".join(f"{int(L)} frames" for L in sorted(acc["length_tag"].unique()) if L > 0)
        + " |"
    )
    lines.append("|---|---|" + "---|" * len([L for L in acc["length_tag"].unique() if L > 0]))
    for noise, sub in acc.groupby("noise"):
        row = [f"{noise:g}", f"{float(full.get(noise, float('nan'))):.3f}"]
        for L in sorted(acc["length_tag"].unique()):
            if L > 0:
                v = sub[sub["length_tag"] == L]["accuracy"]
                row.append(f"{float(v.iloc[0]):.3f}" if len(v) else "")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        f"Accuracy target {lim['accuracy_target']:g}. Shortest length reaching the target, by noise: "
        + ", ".join(
            f"noise {k}: {v if v is not None else 'none'} frames"
            for k, v in lim["shortest_length_frames_by_noise"].items()
        )
        + ".",
        "Highest noise reaching the target, by length: "
        + ", ".join(
            f"{k} frames: {v if v is not None else 'none'}"
            for k, v in lim["highest_noise_by_length_frames"].items()
        )
        + ".",
        "",
    ]
    if len(pooled):
        lines += ["## Posterior law accuracy of one model fitted across the three laws", ""]
        for _, r in pooled.iterrows():
            lines.append(
                f"- noise {r['noise']:g}, {r['law'].replace('_', ' ')}: {r['posterior_accuracy']:.3f} of held-out frames assigned to the true law"
            )
        lines.append("")
    lines += [
        "## Switching detection",
        "",
        f"Detection threshold: the {summary['settings']['detection_level']:.2f} quantile of the held-out switching gap on the null, which is {', '.join(summary.get('null_prefixes', []))}.",
        "",
    ]
    for k, v in lim["detection_by_switch_kind"].items():
        kind, noise = k.split("@")
        lines.append(f"- {kind.replace('_', ' ')} at noise {noise}: detected in {v:.2f} of held-out chases")
    lines.append("")
    lines.append(
        "False detection on misspecified targets outside the null (evader and jumper): "
        + ", ".join(f"noise {k}: {v:.2f}" for k, v in lim["false_detection_misspecified_by_noise"].items())
        + "."
    )
    lines.append(
        f"Detection target {lim['detection_target']:g}; false-detection bound {lim['false_detection_max']:g}. Kinds meeting the target: "
        + ", ".join(lim["switch_kinds_detected_at_target"])
        + "."
    )
    lines.append("")
    if sw:
        lines += [
            "## Switch timing",
            "",
            f"Scripted lead-to-pure switch: median absolute error {_num(sw['median_abs_error_frames'], '.0f')} frames, {_num(sw['within_5_frames'], '.2f')} of decoded switches within 5 frames, {sw['n_missing']} of {sw['n_decoded'] + sw['n_missing']} held-out chases with no decoded switch.",
            "",
        ]
    if par:
        lines += ["## Parameter recovery", ""]
        for name, r in par.items():
            lines.append(
                f"- {name}: bootstrap-interval coverage {r['coverage']:.2f} over {r['n']} conditions, mean absolute error {r['mean_abs_error']:.3f}"
            )
        dr = summary.get("delay_recovery")
        if dr:
            lines.append(
                f"- delay: exact in {dr['exact']:.2f} of {dr['n']} conditions, within one frame in {dr['within_1']:.2f}"
            )
        lines.append("")
    real = summary.get("realistic")
    if real:
        lines += [
            "## The realistic regime",
            "",
            "Conditions with the per-frame gain, delay, noise correlation, tracking noise and glitches seen in the real recordings (gain 0.05, delay 3 frames).",
            "",
        ]
        for r in real.get("classification_by_prefix_and_length", []):
            tag = "full length" if r["length_tag"] == 0 else f"{int(r['length_tag'])} frames"
            lines.append(
                f"- {r['prefix'].replace('_', ' ')}, {tag}: {r['accuracy']:.2f} of {r['n']} held-out chases classified correctly by their own likelihoods"
            )
        for r in real.get("pooled", []):
            lines.append(
                f"- {r['prefix'].replace('_', ' ')}, one model pooled over the three laws: {r['frame_accuracy']:.2f} of held-out frames and {r['chase_vote_accuracy']:.2f} of chases assigned to the true law; "
                f"recovered gain {r['gain']:.3f}, lead time {r['tau']:.2f} s, N {r['N']:.2f}, delay {r['delay_est']} frames; posterior occupancy "
                + ", ".join(f"{v:.2f}" for v in r["posterior_occupancy"])
                + " against a true mix of "
                + ", ".join(f"{v:.2f}" for v in r["true_law_mix"])
            )
        sd = real.get("switching_detection", {})
        for k, v in sd.get("by_prefix", {}).items():
            lines.append(
                f"- {k.replace('_', ' ')}: switching detected in {v['detected_fraction']:.2f} of {v['n']} held-out chases against the realistic single-law null (threshold {sd['threshold']:.4f} per frame)"
            )
        lines.append("")
    lines += [
        "## Calibration",
        "",
        f"Weighted absolute gap between posterior probability and empirical frequency: {summary['calibration_weighted_gap']:.3f}; largest gap in any bin {summary['calibration_max_gap']:.3f}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def realistic_summary(
    suite_dir: Path = SUITE_DIR, out_dir: Path = OUT_DIR, config_path: Path = CONFIG, verbose: bool = True
) -> dict:
    """The realistic regime: pooled fits across the three laws (as on real data), per-chase and
    pooled accuracy, and switching detection against a null drawn from the realistic single-law
    conditions. Computed from the saved per-chase results plus a few pooled fits, and merged into
    summary.json and limits.md."""
    full, light, rc = load_settings(config_path)
    delays = rc["delays"]
    frames, meta = load_suite(suite_dir)
    cls = pd.read_csv(out_dir / "classification.csv").assign(
        prefix=lambda d: d["condition"].str.split("__").str[0]
    )
    out = {}
    # per-chase classification in the realistic regime
    r = cls[cls["group"] == "realistic"].assign(correct=lambda d: d["true_law"] == d["pred_law"])
    out["classification_by_prefix_and_length"] = (
        r.groupby(["prefix", "length_tag"])["correct"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "size": "n"})
        .to_dict(orient="records")
    )
    # switching detection against the realistic null
    level = float(rc["detection_level"])
    full_len = cls[cls["length_tag"] == 0]
    null_gaps = full_len[full_len["group"] == "realistic"]["gap_per_frame"]
    thr = float(null_gaps.quantile(level)) if len(null_gaps) else float("nan")
    det = {}
    for prefix, sub in full_len[full_len["group"] == "realistic_switching"].groupby("prefix"):
        det[prefix] = {"detected_fraction": float((sub["gap_per_frame"] > thr).mean()), "n": len(sub)}
    out["switching_detection"] = {"threshold": thr, "null_n": len(null_gaps), "by_prefix": det}
    # pooled fits: one switching model across the three laws of each realistic prefix
    chases = chases_from_suite(frames, meta)
    by_id = {c.meta.chase_id: c for c in chases}
    pooled = []
    for prefix in ("realistic_single", "realistic_still"):
        m = meta[meta["condition"].str.startswith(prefix + "__")]
        if m.empty or len(set(m["law"])) < 2:
            continue
        train = [by_id[i] for i in m[m["replicate"] % 2 == 0]["chase_id"]]
        test = [by_id[i] for i in m[m["replicate"] % 2 == 1]["chase_id"]]
        if verbose:
            print(f"[realistic pooled] {prefix}: train {len(train)} test {len(test)}", flush=True)
        prof = delay_profile(train, delays, light, model="switching:plain")
        d = best_delay(prof)
        sw = fit_switching(train, replace(full, delay=d), transitions="plain")
        post = posteriors(test, sw)
        truth = {
            c.meta.chase_id: str(meta.loc[meta["chase_id"] == c.meta.chase_id, "law"].iloc[0]) for c in test
        }
        accs, chase_votes = [], []
        for c, pp in zip(test, post, strict=True):
            pred = np.argmax(pp["gamma"], axis=1)
            accs.append(float(np.mean(pred == pp["law_true"].astype(int))))
            chase_votes.append(LAWS[int(np.bincount(pred, minlength=3).argmax())] == truth[c.meta.chase_id])
        occupancy = np.mean(np.concatenate([pp["gamma"] for pp in post]), axis=0)
        true_mix = np.array([np.mean([truth[c.meta.chase_id] == law for c in test]) for law in LAWS])
        pooled.append(
            {
                "prefix": prefix,
                "n_train": len(train),
                "n_test": len(test),
                "delay_est": d,
                "gain": sw.emissions.gain,
                "tau": sw.emissions.tau,
                "N": sw.emissions.N,
                "sigma": sw.emissions.sigma,
                "rho": sw.emissions.rho,
                "eps": sw.emissions.eps,
                "frame_accuracy": float(np.mean(accs)),
                "chase_vote_accuracy": float(np.mean(chase_votes)),
                "posterior_occupancy": occupancy.tolist(),
                "true_law_mix": true_mix.tolist(),
                "stay_probability": np.diag(sw.A).tolist(),
            }
        )
    out["pooled"] = pooled
    with open(out_dir / "realistic.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    sp = out_dir / "summary.json"
    if sp.exists():
        with open(sp, encoding="utf-8") as f:
            summary = json.load(f)
        summary["realistic"] = out
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, default=float)
        write_limits_md(summary, out_dir / "limits.md")
    return out
