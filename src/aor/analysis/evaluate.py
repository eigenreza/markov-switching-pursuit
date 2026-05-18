"""Evaluation on the held-out chases: the optimal-play benchmark and the two-player test, run
from the fitted posteriors written by the fitting pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..contracts import Chase
from . import optimal, twoplayer
from .datasets import DERIVED_DIR, load_real
from .fit_real import RESULTS as FIT_RESULTS
from .fit_real import load_cfg, settings_from_cfg

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "evaluate"

DATASETS = {"michaiel2020": ("dev", "held"), "holmgren2021": ("replication", None), "mabe22": ("dev", "held")}


def held_chases(dataset: str) -> tuple[list[Chase], str]:
    dev_split, held_split = DATASETS[dataset]
    split = held_split or dev_split
    return load_real(dataset, split), split


def run_dataset(
    dataset: str, cfg: dict | None = None, out_dir: Path | None = None, verbose: bool = True
) -> dict:
    cfg = cfg or load_cfg()
    full, _, _ = settings_from_cfg(cfg)
    rc = cfg.get("real", {})
    mp = full.manoeuvre
    out_dir = out_dir or (RESULTS / dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    with open(FIT_RESULTS / dataset / "fit_report.json", encoding="utf-8") as f:
        fit_report = json.load(f)
    delay = int(fit_report["delay_frames"])
    chases, split = held_chases(dataset)
    frames_post = pd.read_parquet(DERIVED_DIR / dataset / f"posteriors_frames_{split}.parquet")
    report = {"dataset": dataset, "split": split, "delay_frames": delay, "n_chases": len(chases)}

    # optimal-play benchmark
    log(f"[{dataset}] optimal-play benchmark on {len(chases)} chases")
    dist, cap, prof = optimal.run(
        chases,
        delay,
        mp.vel_window,
        float(rc.get("informative_speed", 3.0)),
        float(rc.get("contact_radius", 3.0)),
        int(rc.get("seed", 0)),
    )
    dist.to_csv(out_dir / "optimal_distance.csv", index=False)
    cap.to_csv(out_dir / "optimal_capture.csv", index=False)
    prof.to_csv(out_dir / "optimal_profile.csv", index=False)
    d = dist.dropna(subset=["distance_rad"])
    contact = cap[cap["end_reason"] == "contact"].dropna(subset=["optimal_capture_s"])
    report["optimal"] = {
        "distance_rad_median": float(d["distance_rad"].median()),
        "distance_rad_q25": float(d["distance_rad"].quantile(0.25)),
        "distance_rad_q75": float(d["distance_rad"].quantile(0.75)),
        "distance_informative_rad_median": float(d["distance_informative_rad"].median()),
        "distance_to_pure_pursuit_rad_median": float(d["distance_to_pure_pursuit_rad"].median()),
        "distance_by_animal": d.groupby("animal_id")["distance_rad"].median().to_dict(),
        "n_contact_chases": len(contact),
        "optimal_captured_within_chase_fraction": float(cap["optimal_captured_within_chase"].mean()),
        "capture_time_ratio_actual_over_optimal_median": float(
            (contact["actual_capture_s"] / contact["optimal_capture_s"]).median()
        )
        if len(contact)
        else None,
        "capture_time_actual_median_s": float(contact["actual_capture_s"].median()) if len(contact) else None,
        "capture_time_optimal_median_s": float(contact["optimal_capture_s"].median())
        if len(contact)
        else None,
    }

    # two-player test, part 1: target manoeuvres against the pursuer's state
    log(f"[{dataset}] manoeuvre logistic test")
    rows = twoplayer.manoeuvre_rows(chases, frames_post, mp)
    report["manoeuvre_test"] = twoplayer.manoeuvre_test(
        rows,
        B=int(rc.get("manoeuvre_bootstrap", 500)),
        n_perm=int(rc.get("manoeuvre_permutations", 500)),
        seed=int(rc.get("seed", 0)),
    )

    # two-player test, parts 2 and 3: the evader library (mouse targets only)
    if all("t_heading" in c.frames.columns for c in chases) and chases[0].meta.target_type == "mouse":
        log(f"[{dataset}] evader library")
        ev_frames, ev_fit = twoplayer.evader_posteriors(chases, full, mp, delay)
        ev_frames.to_parquet(DERIVED_DIR / dataset / f"evader_frames_{split}.parquet", index=False)
        xc = twoplayer.cross_correlation(
            frames_post,
            ev_frames,
            n_shuffle=int(rc.get("cross_correlation_shuffles", 300)),
            seed=int(rc.get("seed", 0)),
        )
        xc.to_csv(out_dir / "cross_correlation.csv", index=False)
        jo = twoplayer.joint_occupancy(frames_post, ev_frames)
        jo.to_csv(out_dir / "joint_occupancy.csv", index=False)
        peak = xc.iloc[xc["correlation"].abs().argmax()]
        report["evader"] = {
            "fit": {k: ev_fit[k] for k in ("tau", "gain", "N", "sigma", "pi", "A", "loglik", "n_frames")},
            "cross_correlation_peak": {
                "lag_frames": int(peak["lag_frames"]),
                "correlation": float(peak["correlation"]),
                "null_lo": float(peak["null_lo"]),
                "null_hi": float(peak["null_hi"]),
            },
            "cross_correlation_zero_lag": float(xc.loc[xc["lag_frames"] == 0, "correlation"].iloc[0]),
            "outside_null_lags": [
                int(r["lag_frames"])
                for _, r in xc.iterrows()
                if r["correlation"] > r["null_hi"] or r["correlation"] < r["null_lo"]
            ],
            "joint_occupancy_overall": {
                f"{r['pursuer_mode']}|{r['evader_mode']}": float(r["fraction"])
                for _, r in jo[jo["phase_bin"] == -1].iterrows()
            },
        }
    with open(out_dir / "evaluate_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=float)
    log(f"[{dataset}] done")
    return report
