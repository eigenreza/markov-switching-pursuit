"""One style for every figure: a colourblind-safe palette, consistent sizes, and a saver that
writes PDF and 200 dpi PNG without creation or modification metadata."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "results" / "figures"

# Okabe and Ito palette
LAW_COLOURS = {"pure_pursuit": "#D55E00", "lead_pursuit": "#0072B2", "proportional_navigation": "#009E73"}
LAW_LABELS = {
    "pure_pursuit": "pure pursuit",
    "lead_pursuit": "lead pursuit",
    "proportional_navigation": "proportional navigation",
}
TARGET_COLOUR = "#000000"
PURSUER_COLOUR = "#999999"
ACCENT = "#CC79A7"
GREY = "#7F7F7F"

RC = {
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    plt.rcParams.update(RC)


def save(fig, name: str, out_dir: Path = FIG_DIR) -> list[Path]:
    """Write <name>.pdf and <name>.png with metadata fields cleared."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None, "Title": name},
    )
    fig.savefig(png, bbox_inches="tight", metadata={"Software": None, "Title": name})
    plt.close(fig)
    return [pdf, png]


def panel_label(ax, text: str) -> None:
    ax.text(-0.2, 1.08, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
