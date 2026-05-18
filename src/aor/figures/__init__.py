"""Figure generation from committed result files. ``make_all`` writes every figure whose
inputs exist and reports the ones it skipped."""

from __future__ import annotations

from pathlib import Path

from . import fig1_laws, fig2_examples, fig3_recovery, real
from .style import FIG_DIR

FIGURES = {
    "fig1_laws": (fig1_laws.make, []),
    "fig2_examples": (fig2_examples.make, [fig2_examples.EXAMPLES]),
    "fig3_recovery": (
        fig3_recovery.make,
        [fig3_recovery.REC / "classification.csv", fig3_recovery.REC / "summary.json"],
    ),
    "fig4_real_examples": (real.fig4_real_examples, [real.FITS / "michaiel2020" / "posteriors_frames.csv"]),
    "fig5_model_comparison": (
        real.fig5_model_comparison,
        [real.FITS / "michaiel2020" / "comparison_held.csv"],
    ),
    "fig6_event_aligned": (real.fig6_event_aligned, [real.FITS / "michaiel2020" / "event_aligned.csv"]),
    "fig7_parameters": (real.fig7_parameters, [real.FITS / "michaiel2020" / "fit_report.json"]),
    "fig8_optimal": (real.fig8_optimal, [real.EVAL / "michaiel2020" / "optimal_distance.csv"]),
    "fig9_twoplayer": (real.fig9_twoplayer, [real.EVAL / "michaiel2020" / "evaluate_report.json"]),
    "fig10_datasets": (real.fig10_datasets, [real.FITS / "michaiel2020" / "fit_report.json"]),
}


def make_all(out_dir: Path = FIG_DIR) -> dict:
    written, skipped = {}, {}
    for name, (fn, inputs) in FIGURES.items():
        missing = [str(p) for p in inputs if not Path(p).exists()]
        if missing:
            skipped[name] = missing
            continue
        written[name] = [str(p) for p in fn()]
    return {"written": written, "skipped": skipped}
