"""Figures 4 to 10, drawn from the committed result files under results/fits and results/evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from ..laws import LAWS, collision_course_heading, lead_point, los_angle
from .style import ACCENT, GREY, LAW_COLOURS, LAW_LABELS, TARGET_COLOUR, apply_style, panel_label, save

ROOT = Path(__file__).resolve().parents[3]
FITS = ROOT / "results" / "fits"
EVAL = ROOT / "results" / "evaluate"
DATASETS = ["michaiel2020", "holmgren2021", "mabe22"]
DATASET_LABELS = {
    "michaiel2020": "mouse and cricket (primary)",
    "holmgren2021": "mouse and cricket (replication)",
    "mabe22": "mouse and mouse",
}
SHORT_LABELS = {"michaiel2020": "cricket, primary", "holmgren2021": "cricket, replication", "mabe22": "mouse"}
MODEL_LABELS = {
    "null": "persistence null",
    "single:pure_pursuit": "pure pursuit",
    "single:lead_pursuit": "lead pursuit",
    "single:proportional_navigation": "prop. navigation",
    "switching:plain": "switching",
    "switching:input": "switching, input-dependent",
    "switching_ar": "generic switching AR",
}


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False, na_values=[""])  # keep the model name "null"
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=columns or [])


def _report(dataset: str, kind: str = "fit") -> dict:
    p = (FITS if kind == "fit" else EVAL) / dataset / (f"{kind}_report.json")
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _available(kind: str = "fit") -> list[str]:
    return [d for d in DATASETS if ((FITS if kind == "fit" else EVAL) / d / f"{kind}_report.json").exists()]


# ----------------------------------------------------------------------------- figure 4


def fig4_real_examples(dataset: str = "michaiel2020", n: int = 3, out_dir=None):
    """Representative real chases: trajectory coloured by the decoded law, with range and target
    speed traces and the posterior underneath."""
    apply_style()
    frames = _read(FITS / dataset / "posteriors_frames.csv")
    summ = _read(FITS / dataset / "posteriors_chase.csv")
    if frames.empty or summ.empty:
        return []
    cand = summ[(summ["n_frames"] >= 45) & (summ["n_informative"] >= 8)].sort_values(
        "n_informative", ascending=False
    )
    cand = cand[cand["chase_id"].isin(frames["chase_id"].unique())]
    ids = cand["chase_id"].head(n).tolist()
    if not ids:
        ids = summ.sort_values("n_frames", ascending=False)["chase_id"].head(n).tolist()
    fig, axes = plt.subplots(
        3, len(ids), figsize=(2.7 * len(ids) + 0.6, 6.4), gridspec_kw={"height_ratios": [2.3, 1.1, 1.0]}
    )
    axes = np.array(axes).reshape(3, len(ids))
    for j, cid in enumerate(ids):
        g = frames[frames["chase_id"] == cid].sort_values("frame")
        ax = axes[0, j]
        ax.plot(g["tx"], g["ty"], "-", color=TARGET_COLOUR, lw=1.0, label="target")
        ax.plot(g["tx"].iloc[0], g["ty"].iloc[0], "o", color=TARGET_COLOUR, ms=4)
        pts = np.column_stack([g["px"], g["py"]])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        cols = [LAW_COLOURS[LAWS[int(k)]] for k in g["path"].to_numpy()[:-1]]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.0))
        ax.plot(pts[0, 0], pts[0, 1], "s", color=GREY, ms=5)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xlabel("x (cm)")
        if j == 0:
            ax.set_ylabel("y (cm)")
        ax.set_title(f"chase {j + 1}", loc="right", fontsize=8)
        panel_label(ax, "ABC"[j])
        t = g["t"].to_numpy()
        ax2 = axes[1, j]
        ax2.plot(t, g["range"], "-", color=GREY, lw=1.2, label="range (cm)")
        ax2.set_ylabel("range (cm)" if j == 0 else "")
        ax2b = ax2.twinx()
        ax2b.plot(t, g["t_speed"], "-", color=ACCENT, lw=1.0)
        ax2b.set_ylabel("target speed (cm/s)" if j == len(ids) - 1 else "", color=ACCENT)
        ax2b.tick_params(axis="y", colors=ACCENT)
        ax2b.spines["right"].set_visible(True)
        ax3 = axes[2, j]
        ax3.stackplot(
            t,
            g[["p_pure", "p_lead", "p_pn"]].to_numpy().T,
            colors=[LAW_COLOURS[k] for k in LAWS],
            labels=[LAW_LABELS[k] for k in LAWS],
            alpha=0.9,
        )
        ax3.set_ylim(0, 1)
        ax3.set_xlim(t[0], t[-1])
        ax3.set_xlabel("time (s)")
        if j == 0:
            ax3.set_ylabel("posterior")
        if j == len(ids) - 1:
            handles, labels = ax3.get_legend_handles_labels()
            roles = (
                _report(dataset).get("state_roles")
                or _report(dataset).get("posteriors", {}).get("state_roles")
                or []
            )
            labels = [f"{lab} ({roles[k]})" if k < len(roles) else lab for k, lab in enumerate(labels)]
            fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(DATASET_LABELS[dataset], fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return save(fig, "fig4_real_examples", out_dir) if out_dir else save(fig, "fig4_real_examples")


# ----------------------------------------------------------------------------- figure 5


def fig5_model_comparison(out_dir=None):
    """Held-out log-likelihood per frame relative to the best single law, with bootstrap intervals,
    on all frames and on informative frames, for every dataset."""
    apply_style()
    ds = _available("fit")
    if not ds:
        return []
    fig, axes = plt.subplots(1, len(ds), figsize=(2.9 * len(ds) + 0.8, 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    order = [
        "null",
        "single:pure_pursuit",
        "single:lead_pursuit",
        "single:proportional_navigation",
        "switching:plain",
        "switching:input",
        "switching_ar",
    ]
    for ax, d in zip(axes, ds, strict=True):
        comp = _read(FITS / d / "comparison_held.csv").set_index("model")
        inf = _read(FITS / d / "comparison_held_informative.csv")
        inf = inf.set_index("model") if len(inf) else None
        ys = np.arange(len(order))
        for i, m in enumerate(order):
            if m not in comp.index:
                continue
            r = comp.loc[m]
            ax.plot([r["diff_lo"], r["diff_hi"]], [i + 0.12, i + 0.12], "-", color=GREY, lw=1.2)
            ax.plot(
                r["diff_vs_best_single_per_frame"],
                i + 0.12,
                "o",
                color="#000000",
                ms=4,
                label="all frames" if i == 0 else None,
            )
            if inf is not None and m in inf.index:
                q = inf.loc[m]
                ax.plot([q["diff_lo"], q["diff_hi"]], [i - 0.12, i - 0.12], "-", color=ACCENT, lw=1.2)
                ax.plot(
                    q["diff_vs_best_single_per_frame"],
                    i - 0.12,
                    "o",
                    color=ACCENT,
                    ms=4,
                    label="target moving" if i == 0 else None,
                )
        ax.axvline(0, color=GREY, lw=0.6)
        ax.set_yticks(ys, [MODEL_LABELS[m] for m in order])
        ax.set_xlabel("held-out log-likelihood per frame\nminus the best single law")
        ax.set_title(SHORT_LABELS[d], fontsize=9)
        if ax is axes[0]:
            ax.legend(frameon=False, loc="upper right", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
    fig.tight_layout()
    return save(fig, "fig5_model_comparison", out_dir) if out_dir else save(fig, "fig5_model_comparison")


# ----------------------------------------------------------------------------- figure 6


def fig6_event_aligned(out_dir=None):
    apply_style()
    ds = _available("fit")
    if not ds:
        return []
    fig, axes = plt.subplots(1, len(ds), figsize=(2.9 * len(ds) + 0.6, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds, strict=True):
        ev = _read(FITS / d / "event_aligned.csv")
        rep = _report(d)
        fps = float(rep.get("dev", {}).get("fps", 30.0) or 30.0)
        if ev.empty or ev["mean"].isna().all():
            ax.text(0.5, 0.5, "no manoeuvres", ha="center", va="center", transform=ax.transAxes)
            continue
        t = ev["lag_frames"] / fps
        ax.fill_between(t, ev["lo"], ev["hi"], color=LAW_COLOURS["lead_pursuit"], alpha=0.25, lw=0)
        ax.plot(t, ev["mean"], "-", color=LAW_COLOURS["lead_pursuit"], lw=1.5, label="observed")
        if "phase_only" in ev.columns:
            ax.plot(t, ev["phase_only"], "--", color=GREY, lw=1.0, label="chase phase alone")
        ax.axvline(0, color=GREY, lw=0.8, ls=":")
        ax.set_xlabel("time from target manoeuvre (s)")
        ax.set_title(f"{SHORT_LABELS[d]} (n = {int(ev['n_chases'].iloc[0])} chases)", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("posterior steering probability")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    return save(fig, "fig6_event_aligned", out_dir) if out_dir else save(fig, "fig6_event_aligned")


# ----------------------------------------------------------------------------- figure 7


def fig7_parameters(out_dir=None):
    """Fitted parameters with bootstrap intervals per dataset, per-animal steering probability, and the
    lead time and navigation constant across bins of target speed and range."""
    apply_style()
    ds = _available("fit")
    if not ds:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    # (A) parameters
    ax = axes[0]
    names = ["gain", "tau", "N", "delay_ms"]
    labels = {
        "gain": "gain (per frame)",
        "tau": "lead time (s)",
        "N": "navigation constant",
        "delay_ms": "delay (ms)",
    }
    scales = {"gain": 10.0, "tau": 1.0, "N": 0.2, "delay_ms": 0.01}
    for j, d in enumerate(ds):
        rep = _report(d)
        bs = rep.get("bootstrap", {})
        for i, nm in enumerate(names):
            if nm == "delay_ms":
                est, lo, hi = rep.get("delay_ms", np.nan), np.nan, np.nan
            else:
                est, lo, hi = (
                    bs.get(nm, {}).get("estimate", np.nan),
                    bs.get(nm, {}).get("lo", np.nan),
                    bs.get(nm, {}).get("hi", np.nan),
                )
            y = i + (j - (len(ds) - 1) / 2) * 0.22
            if np.isfinite(lo):
                ax.plot([lo * scales[nm], hi * scales[nm]], [y, y], "-", color=GREY, lw=1.0)
            ax.plot(
                est * scales[nm],
                y,
                "o",
                color=[
                    LAW_COLOURS["pure_pursuit"],
                    LAW_COLOURS["lead_pursuit"],
                    LAW_COLOURS["proportional_navigation"],
                ][j % 3],
                ms=4,
                label=SHORT_LABELS[d] if i == 0 else None,
            )
    ax.set_yticks(range(len(names)), [f"{labels[n]} x {scales[n]:g}" for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("estimate (scaled) with bootstrap interval")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "A")
    # (B) per animal steering probability
    ax = axes[1]
    for j, d in enumerate(ds):
        an = _read(FITS / d / "posteriors_animal.csv")
        if an.empty:
            continue
        x = np.full(len(an), j) + np.linspace(-0.15, 0.15, len(an))
        ax.plot(x, an["p_steering_informative"], "o", color=GREY, ms=4, alpha=0.8)
        ax.plot([j - 0.25, j + 0.25], [an["p_steering_informative"].mean()] * 2, "-", color="#000000", lw=1.5)
    ax.set_xticks(range(len(ds)), [SHORT_LABELS[d] for d in ds], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("steering probability per animal\n(target moving)")
    panel_label(ax, "B")
    # (C) tau and N by bins
    ax = axes[2]
    for j, d in enumerate(ds):
        pb = _read(FITS / d / "params_by_bin.csv")
        if pb.empty:
            continue
        for var, marker in (("t_speed", "o"), ("range", "s")):
            sub = pb[pb["variable"] == var].sort_values("bin")
            if sub.empty:
                continue
            x = sub["bin"] + (j - 1) * 0.15
            ax.errorbar(
                x,
                sub["tau"],
                yerr=[sub["tau"] - sub["tau_lo"], sub["tau_hi"] - sub["tau"]],
                fmt=marker,
                color=[
                    LAW_COLOURS["pure_pursuit"],
                    LAW_COLOURS["lead_pursuit"],
                    LAW_COLOURS["proportional_navigation"],
                ][j % 3],
                ms=3,
                elinewidth=0.6,
                capsize=0,
                alpha=0.9 if var == "t_speed" else 0.5,
            )
    ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
    ax.set_xlabel("tertile of target speed (circles) or range (squares)")
    ax.set_ylabel("lead time (s)")
    panel_label(ax, "C")
    fig.tight_layout()
    return save(fig, "fig7_parameters", out_dir) if out_dir else save(fig, "fig7_parameters")


# ----------------------------------------------------------------------------- figure 8


def fig8_optimal(out_dir=None):
    apply_style()
    ds = _available("evaluate")
    if not ds:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
    ax = axes[0]
    for j, d in enumerate(ds):
        dist = _read(EVAL / d / "optimal_distance.csv")
        if dist.empty:
            continue
        v = dist.dropna(subset=["distance_rad"]).sort_values("distance_rad")
        x = np.linspace(0, 1, len(v))
        ax.fill_between(
            x,
            np.degrees(v["distance_lo"]),
            np.degrees(v["distance_hi"]),
            color=LAW_COLOURS[LAWS[j % 3]],
            alpha=0.2,
            lw=0,
        )
        ax.plot(
            x,
            np.degrees(v["distance_rad"]),
            "-",
            color=LAW_COLOURS[LAWS[j % 3]],
            lw=1.2,
            label=SHORT_LABELS[d],
        )
    ax.set_xlabel("chases, sorted")
    ax.set_ylabel("distance from optimal play (deg)")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "A")
    ax = axes[1]
    for j, d in enumerate(ds):
        cap = _read(EVAL / d / "optimal_capture.csv")
        c = cap[(cap["end_reason"] == "contact")].dropna(subset=["optimal_capture_s", "actual_capture_s"])
        if c.empty:
            continue
        ax.plot(
            c["optimal_capture_s"],
            c["actual_capture_s"],
            "o",
            color=LAW_COLOURS[LAWS[j % 3]],
            ms=3,
            alpha=0.7,
            label=f"{SHORT_LABELS[d]} (n = {len(c)})",
        )
    lim = ax.get_xlim()
    hi = max(lim[1], ax.get_ylim()[1])
    ax.plot([0, hi], [0, hi], "--", color=GREY, lw=0.8)
    ax.set_xlabel("capture time of the optimal pursuer (s)")
    ax.set_ylabel("observed capture time (s)")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "B")
    # (C) inset geometry of the optimal strategies
    ax = axes[2]
    px, py, tx, ty, tvx, tvy, vp = 0.0, 0.0, 16.0, 7.0, -2.0, 6.0, 18.0
    lam = float(los_angle(px, py, tx, ty))
    cc = float(collision_course_heading(px, py, tx, ty, tvx, tvy, vp))
    ax.plot([tx], [ty], "o", color=TARGET_COLOUR, ms=6)
    ax.annotate("", xy=(tx + tvx, ty + tvy), xytext=(tx, ty), arrowprops=dict(arrowstyle="->", color=GREY))
    ax.annotate(
        "",
        xy=(tx + 6 * np.cos(lam), ty + 6 * np.sin(lam)),
        xytext=(tx, ty),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6),
    )
    ax.text(
        tx + 6.5 * np.cos(lam),
        ty + 6.5 * np.sin(lam),
        "evader: straight\naway along the\nline of sight",
        fontsize=6.5,
        color=ACCENT,
        va="bottom",
    )
    ax.plot([px], [py], "s", color=GREY, ms=6)
    ax.annotate(
        "",
        xy=(12 * np.cos(cc), 12 * np.sin(cc)),
        xytext=(px, py),
        arrowprops=dict(arrowstyle="-|>", color=LAW_COLOURS["proportional_navigation"], lw=1.8),
    )
    ax.text(
        12.5 * np.cos(cc),
        12.5 * np.sin(cc) - 1.5,
        "pursuer:\nconstant bearing",
        fontsize=6.5,
        color=LAW_COLOURS["proportional_navigation"],
    )
    ax.plot([px, tx], [py, ty], ":", color=ACCENT, lw=0.8)
    lx, ly = lead_point(tx, ty, tvx, tvy, 1.0)
    ax.plot([lx], [ly], "o", mfc="white", mec=TARGET_COLOUR, ms=4)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 26)
    ax.set_ylim(-4, 20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("the game's optimal pair", fontsize=8)
    panel_label(ax, "C")
    fig.tight_layout()
    return save(fig, "fig8_optimal", out_dir) if out_dir else save(fig, "fig8_optimal")


# ----------------------------------------------------------------------------- figure 9


def fig9_twoplayer(out_dir=None):
    apply_style()
    ds = _available("evaluate")
    if not ds:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
    ax = axes[0]
    for j, d in enumerate(ds):
        rep = _report(d, "evaluate").get("manoeuvre_test", {})
        rates = rep.get("manoeuvre_rate_per_frame_by_pursuer_state", {})
        fps = float(_report(d).get("dev", {}).get("fps", 30.0) or 30.0)
        for i, mode in enumerate(("steering", "non-steering")):
            r = rates.get(mode, {}).get("rate")
            if r is not None:
                ax.bar(
                    j + (i - 0.5) * 0.35,
                    r * fps,
                    width=0.33,
                    color=LAW_COLOURS["pure_pursuit"] if mode == "steering" else LAW_COLOURS["lead_pursuit"],
                    label=mode if j == 0 else None,
                )
    ax.set_xticks(range(len(ds)), [SHORT_LABELS[d] for d in ds], rotation=15)
    ax.set_ylabel("target manoeuvres per second")
    ax.set_ylim(0, 1.35 * max(ax.get_ylim()[1], 1e-9))
    ax.legend(
        frameon=False, fontsize=7, title="pursuer state, previous frame", title_fontsize=7, loc="upper left"
    )
    panel_label(ax, "A")
    ax = axes[1]
    xc = _read(EVAL / "mabe22" / "cross_correlation.csv")
    if not xc.empty:
        fps = 30.0
        ax.fill_between(
            xc["lag_frames"] / fps,
            xc["null_lo"],
            xc["null_hi"],
            color=GREY,
            alpha=0.25,
            lw=0,
            label="chase-shuffle null",
        )
        ax.plot(
            xc["lag_frames"] / fps, xc["correlation"], "-o", color="#000000", ms=2.5, lw=1.2, label="observed"
        )
        ax.axvline(0, color=GREY, lw=0.6)
        ax.legend(frameon=False, fontsize=7)
    ax.set_xlabel("lag (s), positive: evader follows pursuer")
    ax.set_ylabel("correlation of pursuer steering\nand evader lead evasion")
    panel_label(ax, "B")
    ax = axes[2]
    jo = _read(EVAL / "mabe22" / "joint_occupancy.csv")
    if not jo.empty:
        phases = sorted(jo[jo["phase_bin"] >= 0]["phase_bin"].unique())
        cells = [
            ("steering", "pure_evasion"),
            ("steering", "other"),
            ("non-steering", "pure_evasion"),
            ("non-steering", "other"),
        ]
        colours = ["#D55E00", "#F0A868", "#0072B2", "#8ABFDF"]
        bottom = np.zeros(len(phases))
        for (pm, em), col in zip(cells, colours, strict=True):
            vals = np.array(
                [
                    float(
                        jo[(jo["phase_bin"] == b) & (jo["pursuer_mode"] == pm) & (jo["evader_mode"] == em)][
                            "fraction"
                        ].sum()
                    )
                    for b in phases
                ]
            )
            ax.bar(
                phases,
                vals,
                bottom=bottom,
                color=col,
                width=0.8,
                label=f"pursuer {pm}, evader {em.replace('_', ' ')}",
            )
            bottom += vals
        ax.set_xticks(
            phases,
            [f"{int(100 * b / len(phases))}-{int(100 * (b + 1) / len(phases))}%" for b in phases],
            fontsize=7,
        )
        ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_xlabel("phase of the chase")
    ax.set_ylabel("fraction of frames")
    ax.set_ylim(0, 1)
    panel_label(ax, "C")
    fig.tight_layout()
    return save(fig, "fig9_twoplayer", out_dir) if out_dir else save(fig, "fig9_twoplayer")


# ----------------------------------------------------------------------------- figure 10


def fig10_datasets(out_dir=None):
    """Predator-prey versus conspecific: steering probability, switching evidence, and parameters side by side."""
    apply_style()
    ds = _available("fit")
    if not ds:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
    ax = axes[0]
    for j, d in enumerate(ds):
        summ = _read(FITS / d / "posteriors_chase.csv")
        if summ.empty:
            continue
        v = summ["p_steering_informative"].dropna()
        parts = ax.violinplot([v], positions=[j], showmedians=True, widths=0.7)
        for b in parts["bodies"]:
            b.set_facecolor(LAW_COLOURS[LAWS[j % 3]])
            b.set_alpha(0.5)
    ax.set_xticks(range(len(ds)), [SHORT_LABELS[d] for d in ds], rotation=15)
    ax.set_ylabel("steering probability per chase\n(target moving)")
    ax.set_ylim(0, 1)
    panel_label(ax, "A")
    ax = axes[1]
    for j, d in enumerate(ds):
        comp = _read(FITS / d / "comparison_held.csv").set_index("model")
        for i, (m, col) in enumerate(
            [
                ("switching:plain", LAW_COLOURS["lead_pursuit"]),
                ("switching:input", LAW_COLOURS["proportional_navigation"]),
                ("null", GREY),
            ]
        ):
            if m in comp.index:
                r = comp.loc[m]
                x = j + (i - 1) * 0.25
                ax.plot([x, x], [r["diff_lo"], r["diff_hi"]], "-", color=col, lw=1.2)
                ax.plot(
                    x,
                    r["diff_vs_best_single_per_frame"],
                    "o",
                    color=col,
                    ms=4,
                    label=MODEL_LABELS[m] if j == 0 else None,
                )
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xticks(range(len(ds)), [SHORT_LABELS[d] for d in ds], rotation=15)
    ax.set_ylabel("held-out log-likelihood per frame\nminus the best single law")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "B")
    ax = axes[2]
    for j, d in enumerate(ds):
        rep = _report(d)
        null = rep.get("switching_null", {})
        gaps = np.array(null.get("null_gaps", []))
        if len(gaps):
            ax.plot(
                np.full(len(gaps), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(gaps)),
                gaps,
                ".",
                color=GREY,
                ms=3,
                alpha=0.6,
            )
            ax.plot(
                j,
                null.get("gap_observed_per_frame", np.nan),
                "D",
                color=ACCENT,
                ms=6,
                label="observed" if j == 0 else None,
            )
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xticks(range(len(ds)), [SHORT_LABELS[d] for d in ds], rotation=15)
    ax.set_ylabel("switching gap per frame:\nobserved vs single-law null")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "C")
    fig.tight_layout()
    return save(fig, "fig10_datasets", out_dir) if out_dir else save(fig, "fig10_datasets")
