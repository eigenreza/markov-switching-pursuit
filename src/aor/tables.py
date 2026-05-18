"""Result tables for the paper, written from the committed result files only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT / "results" / "fits"
EVAL = ROOT / "results" / "evaluate"
OUT = ROOT / "results" / "tables"
DATASETS = ["michaiel2020", "holmgren2021", "mabe22"]
LABELS = {
    "michaiel2020": "mouse and cricket (primary)",
    "holmgren2021": "mouse and cricket (replication)",
    "mabe22": "mouse and mouse",
}
MODELS = [
    "null",
    "single:pure_pursuit",
    "single:lead_pursuit",
    "single:proportional_navigation",
    "switching:plain",
    "switching:input",
    "switching_ar",
]


def _report(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False, na_values=[""])  # keep the model name "null"
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                out = floatfmt.format(v)
                cells.append(
                    floatfmt.format(0.0)
                    if out.lstrip("-+").strip("0.") == "" and out.startswith("-")
                    else out
                )
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    written = {}
    # table 1: datasets and splits
    rows = []
    for d in DATASETS:
        rep = _report(FITS / d / "fit_report.json")
        if not rep:
            continue
        for split in ("dev", "held"):
            if split == "held" and rep.get("in_sample_held"):
                continue  # one replication set: the held row would repeat the dev row
            s = rep[split]
            rows.append(
                {
                    "dataset": LABELS[d],
                    "split": rep[f"{split}_split"],
                    "chases": s["n_chases"],
                    "animals": s["n_animals"],
                    "frames": s["n_frames"],
                    "informative_frames": s["n_informative_frames"],
                    "median_duration_s": round(s["duration_s_median"], 2),
                    "fps": round(s["fps"], 1),
                }
            )
    t1 = pd.DataFrame(rows)
    t1.to_csv(OUT / "table1_datasets.csv", index=False)
    written["table1_datasets"] = t1
    # table 2: model comparison on held-out chases
    rows = []
    for d in DATASETS:
        comp = _csv(FITS / d / "comparison_held.csv")
        inf = _csv(FITS / d / "comparison_held_informative.csv")
        if comp.empty:
            continue
        comp = comp.set_index("model")
        inf = inf.set_index("model") if not inf.empty else None
        for m in MODELS:
            if m not in comp.index:
                continue
            r = comp.loc[m]
            row = {
                "dataset": LABELS[d],
                "model": m,
                "per_frame": r["per_frame"],
                "diff_vs_best_single": r["diff_vs_best_single_per_frame"],
                "diff_lo": r["diff_lo"],
                "diff_hi": r["diff_hi"],
            }
            if inf is not None and m in inf.index:
                row.update(
                    {
                        "diff_informative": inf.loc[m, "diff_vs_best_single_per_frame"],
                        "diff_informative_lo": inf.loc[m, "diff_lo"],
                        "diff_informative_hi": inf.loc[m, "diff_hi"],
                    }
                )
            rows.append(row)
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "table2_model_comparison.csv", index=False)
    written["table2_model_comparison"] = t2
    # table 3: parameters with bootstrap intervals and the switching null
    rows = []
    for d in DATASETS:
        rep = _report(FITS / d / "fit_report.json")
        if not rep:
            continue
        bs = rep["bootstrap"]
        row = {
            "dataset": LABELS[d],
            "delay_frames": rep["delay_frames"],
            "delay_ms": round(rep["delay_ms"], 1),
        }
        for nm in ("gain", "tau", "N", "sigma"):
            row[nm] = bs[nm]["estimate"]
            row[f"{nm}_lo"] = bs[nm]["lo"]
            row[f"{nm}_hi"] = bs[nm]["hi"]
        sw = rep["fits"]["switching:plain"]
        row["rho"] = sw["rho"]
        row["outlier_fraction"] = sw["eps"]
        row["stay_probability_mean"] = sum(sw["A"][i][i] for i in range(3)) / 3
        nl = rep["switching_null"]
        row["switching_gap_observed"] = nl["gap_observed_per_frame"]
        row["switching_gap_null_q95"] = nl["null_q95"]
        row["switching_p_value"] = nl["p_value"]
        rows.append(row)
    t3 = pd.DataFrame(rows)
    t3.to_csv(OUT / "table3_parameters.csv", index=False)
    written["table3_parameters"] = t3
    # table 4: posterior summaries
    rows = []
    for d in DATASETS:
        rep = _report(FITS / d / "fit_report.json")
        if not rep:
            continue
        p = rep["posteriors"]
        rows.append(
            {
                "dataset": LABELS[d],
                "p_steering_all_frames": p["p_steering_mean"],
                "p_steering_target_moving": p["p_steering_informative"],
                "frac_steering_path": p["frac_steering_path"],
                "frac_pure": p["frac_pure"],
                "frac_lead": p["frac_lead"],
                "frac_pn": p["frac_pn"],
                "switches_per_chase": p["switches_per_chase"],
                "chases_with_switch": p["chases_with_switch"],
                "state_roles": " ".join(p["state_roles"]),
            }
        )
    t4 = pd.DataFrame(rows)
    t4.to_csv(OUT / "table4_posteriors.csv", index=False)
    written["table4_posteriors"] = t4
    # table 5: optimal play and the two-player test
    rows = []
    for d in DATASETS:
        rep = _report(EVAL / d / "evaluate_report.json")
        if not rep:
            continue
        o = rep["optimal"]
        m = rep["manoeuvre_test"]
        row = {
            "dataset": LABELS[d],
            "distance_from_optimal_deg_median": round(o["distance_rad_median"] * 57.29578, 1),
            "distance_q25_deg": round(o["distance_rad_q25"] * 57.29578, 1),
            "distance_q75_deg": round(o["distance_rad_q75"] * 57.29578, 1),
            "distance_target_moving_deg": round(o["distance_informative_rad_median"] * 57.29578, 1),
            "distance_to_pure_pursuit_deg": round(o["distance_to_pure_pursuit_rad_median"] * 57.29578, 1),
            "contact_chases": o["n_contact_chases"],
            "capture_time_ratio_median": o["capture_time_ratio_actual_over_optimal_median"],
            "manoeuvre_coef_steering": m["coefficients"]["p_steering_prev"]["estimate"],
            "manoeuvre_coef_lo": m["coefficients"]["p_steering_prev"]["lo"],
            "manoeuvre_coef_hi": m["coefficients"]["p_steering_prev"]["hi"],
            "manoeuvre_null_mean": m["permutation_null_mean"],
            "manoeuvre_within_chase_effect": m["within_chase_effect"],
            "manoeuvre_permutation_p": m["permutation_p_steering"],
            "n_manoeuvres": m["n_manoeuvres"],
        }
        ev = rep.get("evader")
        if ev:
            row["cross_correlation_zero_lag"] = ev["cross_correlation_zero_lag"]
            row["cross_correlation_peak"] = ev["cross_correlation_peak"]["correlation"]
            row["cross_correlation_peak_lag_frames"] = ev["cross_correlation_peak"]["lag_frames"]
        rows.append(row)
    t5 = pd.DataFrame(rows)
    t5.to_csv(OUT / "table5_game.csv", index=False)
    written["table5_game"] = t5
    # table 6: the post-freeze follow-up with a persistence state against the frozen analysis
    rows = []
    for d in DATASETS:
        rep3 = _report(FITS / d / "fit_report.json")
        rep4 = _report(ROOT / "results" / "fits_persistence" / d / "fit_report.json")
        if not rep3 or not rep4:
            continue
        for rep, name in ((rep3, "three laws"), (rep4, "three laws and persistence")):
            comp = {r["model"]: r for r in rep["comparison_held"]}
            bs = rep["bootstrap"]
            po = rep["posteriors"]
            rows.append(
                {
                    "dataset": LABELS[d],
                    "library": name,
                    "delay_frames": rep["delay_frames"],
                    "gain": bs["gain"]["estimate"],
                    "tau": bs["tau"]["estimate"],
                    "N": bs["N"]["estimate"],
                    "N_lo": bs["N"]["lo"],
                    "N_hi": bs["N"]["hi"],
                    "switching_gap": comp["switching:plain"]["diff_vs_best_single_per_frame"],
                    "switching_gap_lo": comp["switching:plain"]["diff_lo"],
                    "switching_gap_hi": comp["switching:plain"]["diff_hi"],
                    "comparator_gap": comp["switching_ar"]["diff_vs_best_single_per_frame"],
                    "null_p": rep["switching_null"]["p_value"],
                    "p_steering": po["p_steering_mean"],
                    "frac_pure": po.get("frac_pure"),
                    "frac_pn": po.get("frac_pn"),
                    "frac_persistence": po.get("frac_persistence", 0.0),
                    "state_roles": " ".join(po["state_roles"]),
                }
            )
    t6 = pd.DataFrame(rows)
    t6.to_csv(OUT / "table6_followup.csv", index=False)
    written["table6_followup"] = t6
    # table 7: leave-one-animal-out on the replication set
    t7 = _csv(FITS / "holmgren2021" / "leave_one_animal_out.csv")
    if not t7.empty:
        t7 = t7[
            [
                "frames",
                "model",
                "per_frame",
                "diff_vs_best_single_per_frame",
                "diff_lo",
                "diff_hi",
                "n_chases",
                "n_frames",
            ]
        ]
    t7.to_csv(OUT / "table7_leave_one_animal_out.csv", index=False)
    written["table7_leave_one_animal_out"] = t7
    # table 8: scale sensitivity of the primary dataset
    rows = []
    for name, path in (
        ("deposit scale", FITS / "michaiel2020" / "fit_report.json"),
        ("arena-fit scale", FITS / "michaiel2020_arena_scale" / "fit_report.json"),
    ):
        rep = _report(path)
        if not rep:
            continue
        comp = {r["model"]: r for r in rep["comparison_held"]}
        bs = rep["bootstrap"]
        rows.append(
            {
                "scale": name,
                "dev_chases": rep["dev"]["n_chases"],
                "held_chases": rep["held"]["n_chases"],
                "held_informative_frames": rep["held"]["n_informative_frames"],
                "delay_frames": rep["delay_frames"],
                "gain": bs["gain"]["estimate"],
                "gain_lo": bs["gain"]["lo"],
                "gain_hi": bs["gain"]["hi"],
                "N": bs["N"]["estimate"],
                "switching_gap": comp["switching:plain"]["diff_vs_best_single_per_frame"],
                "switching_gap_lo": comp["switching:plain"]["diff_lo"],
                "switching_gap_hi": comp["switching:plain"]["diff_hi"],
                "comparator_gap": comp["switching_ar"]["diff_vs_best_single_per_frame"],
                "null_p": rep["switching_null"]["p_value"],
                "p_steering": rep["posteriors"]["p_steering_mean"],
            }
        )
    t8 = pd.DataFrame(rows)
    t8.to_csv(OUT / "table8_scale_sensitivity.csv", index=False)
    written["table8_scale_sensitivity"] = t8
    # table 9: what predicts switching: input-dependent against fixed transitions on held-out chases,
    # and the steering probability around target manoeuvres against the chase-phase prediction
    rows = []
    rng = np.random.default_rng(0)
    for d in DATASETS:
        rep = _report(FITS / d / "fit_report.json")
        sc = _csv(FITS / d / "scores_held.csv")
        ev = _csv(FITS / d / "event_aligned.csv")
        if not rep or sc.empty:
            continue
        sc = sc.dropna(subset=["switching:input", "switching:plain"])
        diff = (sc["switching:input"] - sc["switching:plain"]).to_numpy()
        n = sc["n_frames"].to_numpy()
        groups = [np.flatnonzero(sc["animal_id"].to_numpy() == a) for a in sc["animal_id"].unique()]
        boots = []
        for _ in range(1000):
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
            boots.append(diff[idx].sum() / n[idx].sum())
        comp = {r["model"]: r for r in rep["comparison_held"]}
        gap = comp["switching:plain"]["diff_vs_best_single_per_frame"]
        row = {
            "dataset": LABELS[d],
            "input_minus_plain": diff.sum() / n.sum(),
            "input_minus_plain_lo": float(np.percentile(boots, 2.5)),
            "input_minus_plain_hi": float(np.percentile(boots, 97.5)),
            "switching_gap": gap,
            "share_of_switching_gap": (diff.sum() / n.sum()) / gap if gap else np.nan,
        }
        if not ev.empty and "phase_only" in ev.columns and ev["mean"].notna().any():
            t = ev["lag_frames"] / float(rep["dev"]["fps"])
            pre, post = (t >= -0.5) & (t <= -0.1), (t >= 0.1) & (t <= 0.5)
            row.update(
                {
                    "event_chases": int(ev["n_chases"].iloc[0]),
                    "steer_before": float(ev.loc[pre, "mean"].mean()),
                    "steer_after": float(ev.loc[post, "mean"].mean()),
                    "phase_only_before": float(ev.loc[pre, "phase_only"].mean()),
                    "phase_only_after": float(ev.loc[post, "phase_only"].mean()),
                }
            )
            row["steer_change"] = row["steer_after"] - row["steer_before"]
            row["phase_only_change"] = row["phase_only_after"] - row["phase_only_before"]
        rows.append(row)
    t9 = pd.DataFrame(rows)
    t9.to_csv(OUT / "table9_switch_predictors.csv", index=False)
    written["table9_switch_predictors"] = t9
    with open(OUT / "tables.md", "w", encoding="utf-8") as f:
        for name, df in written.items():
            f.write(f"## {name}\n\n{md_table(df) if len(df) else '(no rows)'}\n\n")
    return {k: len(v) for k, v in written.items()}
