"""Figure 2: one simulated chase per law with the inferred law posterior underneath."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from ..laws import LAWS
from .style import GREY, LAW_COLOURS, LAW_LABELS, TARGET_COLOUR, apply_style, panel_label, save

SHORT = {
    "pure_pursuit": "pure pursuit",
    "lead_pursuit": "lead pursuit",
    "proportional_navigation": "prop. navigation",
}

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "results" / "recovery" / "examples.json"


def make(examples_path: Path = EXAMPLES, out_dir=None):
    apply_style()
    with open(examples_path, encoding="utf-8") as f:
        ex = json.load(f)
    laws = [law for law in LAWS if law in ex]
    if not laws:
        return []
    fig, axes = plt.subplots(
        2, len(laws), figsize=(2.6 * len(laws) + 0.4, 4.6), gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    axes = np.array(axes).reshape(2, len(laws))
    for j, law in enumerate(laws):
        e = ex[law]
        ax = axes[0, j]
        px, py, tx, ty = (np.array(e[k]) for k in ("px", "py", "tx", "ty"))
        ax.plot(tx, ty, "-", color=TARGET_COLOUR, lw=1.0, label="target")
        ax.plot(tx[0], ty[0], "o", color=TARGET_COLOUR, ms=4)
        ax.plot(px, py, "-", color=LAW_COLOURS[law], lw=1.6, label="pursuer")
        ax.plot(px[0], py[0], "s", color=GREY, ms=5)
        ax.set_aspect("equal")
        ax.set_title(SHORT[law], color=LAW_COLOURS[law], loc="right")
        ax.set_xlabel("x (cm)")
        if j == 0:
            ax.set_ylabel("y (cm)")
            ax.legend(loc="best", frameon=False)
        panel_label(ax, "ABC"[j])
        ax2 = axes[1, j]
        t = np.array(e["t"])[np.array(e["s"])]
        g = np.array(e["gamma"])
        ax2.stackplot(
            t, g.T, colors=[LAW_COLOURS[k] for k in LAWS], labels=[LAW_LABELS[k] for k in LAWS], alpha=0.9
        )
        ax2.set_ylim(0, 1)
        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel("time (s)")
        if j == 0:
            ax2.set_ylabel("posterior probability")
        if j == len(laws) - 1:
            ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False)
    fig.tight_layout()
    return save(fig, "fig2_examples", out_dir) if out_dir else save(fig, "fig2_examples")
