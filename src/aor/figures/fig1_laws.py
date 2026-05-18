"""Figure 1: the three guidance laws drawn on one geometry, showing where each sends the pursuer."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..laws import collision_course_heading, lead_point, los_angle
from .style import ACCENT, GREY, LAW_COLOURS, LAW_LABELS, TARGET_COLOUR, apply_style, save


def make(out_dir=None):
    apply_style()
    px, py, heading = 0.0, 0.0, np.deg2rad(20.0)
    tx, ty = 18.0, 9.0
    tvx, tvy = -2.0, 7.0  # cm/s, the target crosses from right to left while moving up
    v_p = 20.0
    tau = 0.8
    lam = float(los_angle(px, py, tx, ty))
    lx, ly = lead_point(tx, ty, tvx, tvy, tau)
    lead_angle = float(los_angle(px, py, lx, ly))
    cc = float(collision_course_heading(px, py, tx, ty, tvx, tvy, v_p))

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    # target, its velocity, and the lead point
    ax.plot([tx], [ty], "o", color=TARGET_COLOUR, ms=6, zorder=5)
    ax.annotate(
        "",
        xy=(tx + tvx * tau, ty + tvy * tau),
        xytext=(tx, ty),
        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0),
    )
    ax.plot([lx], [ly], "o", mfc="white", mec=TARGET_COLOUR, ms=6, zorder=5)
    ax.text(tx + 0.8, ty - 1.6, "target now", fontsize=7.5)
    ax.text(lx + 0.8, ly + 0.4, "target after lead time", fontsize=7.5)
    # pursuer and heading
    ax.plot([px], [py], "s", color=GREY, ms=7, zorder=5)
    ax.annotate(
        "",
        xy=(4 * np.cos(heading), 4 * np.sin(heading)),
        xytext=(px, py),
        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4),
    )
    ax.text(0.6, -2.0, "pursuer and its heading", fontsize=7.5, color=GREY)
    # the three commanded directions
    L = 13.0
    for law, ang, dy in [
        ("pure_pursuit", lam, -0.9),
        ("lead_pursuit", lead_angle, -0.6),
        ("proportional_navigation", cc, 0.9),
    ]:
        ax.annotate(
            "",
            xy=(L * np.cos(ang), L * np.sin(ang)),
            xytext=(px, py),
            arrowprops=dict(arrowstyle="-|>", color=LAW_COLOURS[law], lw=1.8),
        )
        ax.text(
            1.04 * L * np.cos(ang) + 0.3,
            1.04 * L * np.sin(ang) + dy,
            LAW_LABELS[law],
            color=LAW_COLOURS[law],
            fontsize=8,
            ha="left",
            va="center",
        )
    # line of sight
    ax.plot([px, tx], [py, ty], ":", color=ACCENT, lw=1.0, zorder=1)
    ax.text(
        0.66 * tx + 0.4,
        0.66 * ty - 2.3,
        "line of sight",
        color=ACCENT,
        fontsize=7.5,
        rotation=np.degrees(lam),
        ha="center",
        va="top",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-3, 24)
    ax.set_ylim(-4, 19)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Where each law sends the pursuer")
    return save(fig, "fig1_laws", out_dir) if out_dir else save(fig, "fig1_laws")
