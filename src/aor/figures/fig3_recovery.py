"""Figure 3: the recovery study. Confusion matrix, switch-time error, parameter recovery with
intervals, and the calibration curve."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..laws import LAWS
from .style import ACCENT, GREY, LAW_COLOURS, LAW_LABELS, apply_style, panel_label, save

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "results" / "recovery"
SHORT = {"pure_pursuit": "pure", "lead_pursuit": "lead", "proportional_navigation": "prop. nav."}


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=columns)


def make(rec_dir: Path = REC, out_dir=None):
    apply_style()
    cls = read_csv(rec_dir / "classification.csv", ["group", "length_tag", "noise", "true_law", "pred_law"])
    sw = read_csv(rec_dir / "switch_time.csv", ["error_frames"])
    par = read_csv(rec_dir / "params.csv", ["parameter", "true", "estimate", "lo", "hi"])
    cal = read_csv(rec_dir / "calibration.csv", ["law", "mean_predicted", "empirical", "n"])
    with open(rec_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.6))

    # (A) confusion matrix, full length, pooled over noise levels up to 0.1
    ax = axes[0, 0]
    single = cls[(cls["group"] == "single") & (cls["length_tag"] == 0) & (cls["noise"] <= 0.1)]
    M = np.zeros((3, 3))
    for i, tl in enumerate(LAWS):
        for j, pl in enumerate(LAWS):
            M[i, j] = ((single["true_law"] == tl) & (single["pred_law"] == pl)).sum()
    Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{Mn[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if Mn[i, j] > 0.6 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(3), [SHORT[k] for k in LAWS])
    ax.set_yticks(range(3), [SHORT[k] for k in LAWS])
    ax.set_xlabel("classified as")
    ax.set_ylabel("generating law")
    ax.set_title(f"law classification (n = {int(M.sum())})")
    ax.spines[["top", "right"]].set_visible(True)
    panel_label(ax, "A")

    # (B) switch-time error
    ax = axes[0, 1]
    err = sw["error_frames"].dropna() / 30.0
    if len(err):
        ax.hist(err, bins=np.arange(-1.0, 1.05, 0.1), color=ACCENT, edgecolor="white")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("decoded minus true switch time (s)")
    ax.set_ylabel("held-out chases")
    missing = int(sw["error_frames"].isna().sum())
    ax.set_title(f"switch timing (n = {len(err)}, undetected {missing})")
    panel_label(ax, "B")

    # (C) parameter recovery
    ax = axes[1, 0]
    offsets = {"tau": 0, "N": 1, "delay": 2}
    xs, ys, los, his, cols = [], [], [], [], []
    for _, r in par.iterrows():
        if r["parameter"] not in offsets:
            continue
        scale = {"tau": 1.0, "N": 1.0 / 8.0, "delay": 1.0 / 6.0}[r["parameter"]]
        xs.append(r["true"] * scale + offsets[r["parameter"]] * 1.3)
        ys.append(r["estimate"] * scale)
        los.append(r["lo"] * scale)
        his.append(r["hi"] * scale)
        cols.append(
            {"tau": LAW_COLOURS["lead_pursuit"], "N": LAW_COLOURS["proportional_navigation"], "delay": GREY}[
                r["parameter"]
            ]
        )
    if xs:
        ax.vlines(xs, los, his, color=GREY, lw=0.8, zorder=2)
        ax.scatter(xs, ys, c=cols, s=14, zorder=3)
        for off in offsets.values():
            xx = np.linspace(0, 1, 10)
            ax.plot(xx + off * 1.3, xx, "-", color=GREY, lw=0.6, zorder=1)
        ax.set_xticks([0.5, 1.8, 3.1], ["lead time (s)", "N / 8", "delay / 6"])
    ax.set_xlabel("true value, by parameter")
    ax.set_ylabel("estimate with bootstrap interval")
    ax.set_ylim(-0.1, 1.1)
    cov = {r["parameter"]: r["coverage"] for r in summary.get("parameter_recovery", [])}
    ax.set_title("parameter recovery")
    ax.text(
        0.98,
        0.04,
        "interval coverage\n" + "\n".join(f"{k} {v:.2f}" for k, v in cov.items()),
        transform=ax.transAxes,
        fontsize=6.5,
        va="bottom",
        ha="right",
    )
    panel_label(ax, "C")

    # (D) calibration
    ax = axes[1, 1]
    for law in LAWS:
        c = cal[cal["law"] == law]
        ax.plot(
            c["mean_predicted"], c["empirical"], "o-", color=LAW_COLOURS[law], ms=3, label=LAW_LABELS[law]
        )
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
    ax.set_xlabel("posterior probability of the law")
    ax.set_ylabel("fraction of frames produced by it")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("calibration of the law posterior")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "D")
    fig.tight_layout()
    return save(fig, "fig3_recovery", out_dir) if out_dir else save(fig, "fig3_recovery")
