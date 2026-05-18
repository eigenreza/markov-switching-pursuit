"""The paper's numbers and tables, written from the committed result files only.

``write_numbers`` turns the result files into LaTeX macros (``paper/numbers.tex``) so that every
number in the text comes from a result file; ``write_tables`` turns the result tables into LaTeX
tables (``paper/tables/``); ``build`` compiles the PDF with latexmk. The consistency test
regenerates both and fails if they differ from the committed files.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FITS = RESULTS / "fits"
FITS_FOLLOWUP = RESULTS / "fits_persistence"
EVAL = RESULTS / "evaluate"
TABLES = RESULTS / "tables"
PAPER = ROOT / "paper"
PREFIX = {"michaiel2020": "Mich", "holmgren2021": "Holm", "mabe22": "Mabe"}
RAD2DEG = 57.29577951308232


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False, na_values=[""])  # keep the model name "null"
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def fmt(v, nd: int = 2, thousands: bool = False) -> str:
    """Fixed-decimal formatting; integers with an optional thousands separator; n/a for missing."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if nd == 0:
        return f"{round(float(v)):,}" if thousands else f"{round(float(v))}"
    out = f"{float(v):.{nd}f}"
    return out[1:] if out.startswith("-") and float(out) == 0 else out  # no negative zero


def _signed(v, nd: int = 3) -> str:
    """Signed fixed-decimal formatting; a value that rounds to zero prints unsigned."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    out = f"{float(v):+.{nd}f}"
    return f"{0:.{nd}f}" if float(out) == 0 else out


class Numbers:
    def __init__(self):
        self.macros: dict[str, str] = {}
        self.missing: list[str] = []

    def add(self, name: str, value: str) -> None:
        assert re.fullmatch(r"[A-Za-z]+", name), name
        self.macros[name] = value

    def tex(self) -> str:
        lines = [
            "% Numbers for the paper, written by `python tasks.py paper` from the result files.",
            "% Do not edit by hand.",
        ]
        for k in sorted(self.macros):
            lines.append(f"\\newcommand{{\\{k}}}{{{self.macros[k]}}}")
        return "\n".join(lines) + "\n"


def _fit_numbers(nums: Numbers, rep: dict, pre: str) -> None:
    if not rep:
        nums.missing.append(pre)
        return
    for split, tag in (("dev", "Dev"), ("held", "Held")):
        s = rep[split]
        nums.add(f"{pre}{tag}Chases", fmt(s["n_chases"], 0, True))
        nums.add(f"{pre}{tag}Animals", fmt(s["n_animals"], 0))
        nums.add(f"{pre}{tag}Frames", fmt(s["n_frames"], 0, True))
        nums.add(f"{pre}{tag}Inf", fmt(s["n_informative_frames"], 0, True))
        nums.add(f"{pre}{tag}InfPct", fmt(100 * s["informative_fraction"], 0))
        nums.add(f"{pre}{tag}DurMedian", fmt(s["duration_s_median"], 2))
    nums.add(f"{pre}Fps", fmt(rep["dev"]["fps"], 1))
    nums.add(f"{pre}DelayFrames", fmt(rep["delay_frames"], 0))
    nums.add(f"{pre}DelayMs", fmt(rep["delay_ms"], 0))
    comp = {r["model"]: r for r in rep.get("comparison_held", [])}
    names = {
        "null": "Null",
        "single:pure_pursuit": "Pure",
        "single:lead_pursuit": "Lead",
        "single:proportional_navigation": "PN",
        "switching:plain": "Switching",
        "switching:input": "Input",
        "switching_ar": "AR",
    }
    for m, tag in names.items():
        if m in comp:
            r = comp[m]
            nums.add(f"{pre}Gap{tag}", _signed(r["diff_vs_best_single_per_frame"], 4))
            nums.add(f"{pre}Gap{tag}Lo", _signed(r["diff_lo"], 4))
            nums.add(f"{pre}Gap{tag}Hi", _signed(r["diff_hi"], 4))
            nums.add(f"{pre}PerFrame{tag}", fmt(r["per_frame"], 3))
            nums.add(f"{pre}Gap{tag}Abs", fmt(abs(r["diff_vs_best_single_per_frame"]), 4))
    nums.add(
        f"{pre}BestSingle",
        comp.get("null", {}).get("best_single", "n/a").replace("single:", "").replace("_", " "),
    )
    nums.add(f"{pre}HeldScoredChases", fmt(comp.get("null", {}).get("n_chases"), 0, True))
    nums.add(f"{pre}HeldScoredFrames", fmt(comp.get("null", {}).get("n_frames"), 0, True))
    inf = {r["model"]: r for r in rep.get("comparison_held_informative", [])}
    for m, tag in names.items():
        if m in inf:
            r = inf[m]
            nums.add(f"{pre}InfGap{tag}", _signed(r["diff_vs_best_single_per_frame"], 4))
            nums.add(f"{pre}InfGap{tag}Lo", _signed(r["diff_lo"], 4))
            nums.add(f"{pre}InfGap{tag}Hi", _signed(r["diff_hi"], 4))
    if inf:
        nums.add(f"{pre}InfScoredFrames", fmt(inf["null"]["n_frames"], 0, True))
        nums.add(f"{pre}InfScoredChases", fmt(inf["null"]["n_chases"], 0, True))
    nl = rep.get("switching_null", {})
    if nl:
        B = int(nl.get("B", len(nl.get("null_gaps", []))))
        floor = 1.0 / (B + 1)
        if B and abs(nl["p_value"] - floor) < 1e-9:
            nums.add(f"{pre}NullP", f"$< {fmt(floor, 3)}$")  # the smallest value B replicates allow
        else:
            nums.add(f"{pre}NullP", f"$= {fmt(nl['p_value'], 3)}$")
        nums.add(f"{pre}NullPValue", fmt(nl["p_value"], 3))
        nums.add(f"{pre}NullQ", _signed(nl["null_q95"], 4))
        nums.add(f"{pre}NullB", fmt(nl.get("B", len(nl.get("null_gaps", []))), 0))
        nums.add(f"{pre}NullLaw", nl["law"].replace("_", " "))
    bs = rep.get("bootstrap", {})
    for name, tag, nd in (("gain", "Gain", 3), ("tau", "Tau", 3), ("N", "Nav", 3), ("sigma", "Sigma", 3)):
        if name in bs:
            nums.add(f"{pre}{tag}", fmt(bs[name]["estimate"], nd))
            nums.add(f"{pre}{tag}Lo", fmt(bs[name]["lo"], nd))
            nums.add(f"{pre}{tag}Hi", fmt(bs[name]["hi"], nd))
    sw = rep.get("fits", {}).get("switching:plain", {})
    if sw:
        nums.add(f"{pre}Rho", fmt(sw["rho"], 2))
        nums.add(f"{pre}Eps", fmt(sw["eps"], 3))
        nums.add(f"{pre}SigmaOut", fmt(sw["sigma_out"], 2))
        nums.add(
            f"{pre}TimeConstant", fmt(1.0 / (sw["gain"] * rep["dev"]["fps"]), 2) if sw["gain"] > 0 else "n/a"
        )
        stay = [sw["A"][i][i] for i in range(len(sw["A"]))]
        nums.add(f"{pre}StayMin", fmt(min(stay), 2))
        nums.add(f"{pre}StayMax", fmt(max(stay), 2))
    single = rep.get("fits", {}).get(rep.get("best_single_dev", ""), {})
    if single:
        nums.add(f"{pre}SingleGain", fmt(single["gain"], 3))
        nums.add(f"{pre}SingleTau", fmt(single["tau"], 3))
        nums.add(f"{pre}SingleNav", fmt(single["N"], 3))
    ar = rep.get("fits", {}).get("switching_ar", {})
    if ar:
        nums.add(f"{pre}ARRho", fmt(ar["rho"], 2))
    po = rep.get("posteriors", {})
    if po and "p_steering_mean" in po:
        nums.add(f"{pre}PSteer", fmt(po["p_steering_mean"], 2))
        nums.add(f"{pre}PSteerInf", fmt(po["p_steering_informative"], 2))
        nums.add(f"{pre}PSteerPct", fmt(100 * po["p_steering_mean"], 0))
        nums.add(f"{pre}SteerPathPct", fmt(100 * po["frac_steering_path"], 0))
        for name in ("pure", "lead", "pn", "persistence"):
            if f"frac_{name}" in po:
                nums.add(f"{pre}Frac{name.capitalize()}Pct", fmt(100 * po[f"frac_{name}"], 0))
        nums.add(f"{pre}SwitchesPerChase", fmt(po["switches_per_chase"], 2))
        nums.add(f"{pre}ChasesWithSwitchPct", fmt(100 * po["chases_with_switch"], 0))
        nums.add(f"{pre}Roles", ", ".join(po["state_roles"]))
    nums.add(f"{pre}EventChases", fmt(rep.get("event_aligned_n_chases"), 0))


def _eval_numbers(nums: Numbers, rep: dict, pre: str) -> None:
    if not rep:
        nums.missing.append(pre + " evaluate")
        return
    o = rep["optimal"]
    nums.add(f"{pre}DistOpt", fmt(o["distance_rad_median"] * RAD2DEG, 0))
    nums.add(f"{pre}DistOptQa", fmt(o["distance_rad_q25"] * RAD2DEG, 0))
    nums.add(f"{pre}DistOptQb", fmt(o["distance_rad_q75"] * RAD2DEG, 0))
    nums.add(f"{pre}DistOptInf", fmt(o["distance_informative_rad_median"] * RAD2DEG, 0))
    nums.add(f"{pre}DistPP", fmt(o["distance_to_pure_pursuit_rad_median"] * RAD2DEG, 0))
    nums.add(f"{pre}ContactChases", fmt(o["n_contact_chases"], 0))
    nums.add(f"{pre}CaptureRatio", fmt(o["capture_time_ratio_actual_over_optimal_median"], 2))
    nums.add(f"{pre}CaptureActual", fmt(o["capture_time_actual_median_s"], 2))
    nums.add(f"{pre}CaptureOptimal", fmt(o["capture_time_optimal_median_s"], 2))
    nums.add(f"{pre}OptCapturedPct", fmt(100 * o["optimal_captured_within_chase_fraction"], 0))
    m = rep["manoeuvre_test"]
    c = m["coefficients"]["p_steering_prev"]
    nums.add(f"{pre}ManCoef", _signed(c["estimate"], 2))
    nums.add(f"{pre}ManCoefLo", _signed(c["lo"], 2))
    nums.add(f"{pre}ManCoefHi", _signed(c["hi"], 2))
    nums.add(f"{pre}ManNullMean", _signed(m["permutation_null_mean"], 2))
    nums.add(f"{pre}ManWithin", _signed(m["within_chase_effect"], 2))
    nums.add(f"{pre}ManP", fmt(m["permutation_p_steering"], 3))
    nums.add(f"{pre}ManN", fmt(m["n_manoeuvres"], 0, True))
    nums.add(f"{pre}ManFrames", fmt(m["n_frames"], 0, True))
    rc = m["coefficients"]["range_per_10cm"]
    nums.add(f"{pre}ManRange", _signed(rc["estimate"], 2))
    rates = m["manoeuvre_rate_per_frame_by_pursuer_state"]
    for state, tag in (("steering", "Steer"), ("non-steering", "NonSteer")):
        if state in rates:
            nums.add(f"{pre}ManRate{tag}", fmt(rates[state]["rate"] * 100, 2))
    ev = rep.get("evader")
    if ev:
        nums.add(f"{pre}EvaderGain", fmt(ev["fit"]["gain"], 4))
        nums.add(f"{pre}EvaderTau", fmt(ev["fit"]["tau"], 2))
        nums.add(f"{pre}XcorrZero", _signed(ev["cross_correlation_zero_lag"], 3))
        nums.add(f"{pre}XcorrPeak", _signed(ev["cross_correlation_peak"]["correlation"], 3))
        nums.add(f"{pre}XcorrPeakLag", fmt(ev["cross_correlation_peak"]["lag_frames"], 0))
        nums.add(f"{pre}XcorrNullLo", _signed(ev["cross_correlation_peak"]["null_lo"], 3))
        nums.add(f"{pre}XcorrNullHi", _signed(ev["cross_correlation_peak"]["null_hi"], 3))
        nums.add(f"{pre}XcorrOutside", fmt(len(ev["outside_null_lags"]), 0))


def _recovery_numbers(nums: Numbers, s: dict) -> None:
    if not s:
        nums.missing.append("recovery")
        return
    cls = {(r["noise"], r["length_tag"]): r["accuracy"] for r in s["classification_accuracy"]}
    noises = sorted({k[0] for k in cls})
    full = [cls[(n, 0)] for n in noises if (n, 0) in cls]
    nums.add("RecNoiseMin", fmt(min(noises), 2))
    nums.add("RecNoiseMax", fmt(max(noises), 2))
    nums.add("RecClassFullMin", fmt(min(full), 2))
    nums.add("RecClassFullMax", fmt(max(full), 2))
    lim = s["limits"]
    nums.add(
        "RecShortestLowNoise",
        fmt(
            lim["shortest_length_frames_by_noise"][str(min(noises))]
            if str(min(noises)) in lim["shortest_length_frames_by_noise"]
            else lim["shortest_length_frames_by_noise"][min(noises)],
            0,
        ),
    )
    sh = lim["shortest_length_frames_by_noise"]
    vals = [v for v in sh.values()]
    nums.add("RecShortestMax", fmt(max(vals), 0))
    nums.add("RecShortestMin", fmt(min(vals), 0))
    det = lim["detection_by_switch_kind"]
    for key, tag in (
        ("switch_fixed", "Fixed"),
        ("switch_markov", "Markov"),
        ("switch_range", "Range"),
        ("switch_manoeuvre_turn", "ManoeuvreTurn"),
        ("switch_manoeuvre", "ManoeuvreStill"),
    ):
        vals = [v for k, v in det.items() if k.startswith(key + "@")]
        if vals:
            nums.add(f"RecDetect{tag}Min", fmt(min(vals), 2))
            nums.add(f"RecDetect{tag}Max", fmt(max(vals), 2))
    fp = lim.get("false_detection_misspecified_by_noise", {})
    if fp:
        nums.add("RecFalseMax", fmt(max(fp.values()), 2))
    for r in s["parameter_recovery"]:
        tag = {"tau": "Tau", "N": "Nav", "gain": "Gain", "delay": "Delay"}.get(
            r["parameter"], r["parameter"].capitalize()
        )
        nums.add(f"RecCoverage{tag}", fmt(r["coverage"], 2))
        nums.add(f"RecMae{tag}", fmt(r["mean_abs_error"], 3))
    dr = s.get("delay_recovery", {})
    if dr:
        nums.add("RecDelayExact", fmt(dr["exact"], 2))
        nums.add("RecDelayN", fmt(dr["n"], 0))
    st = s.get("switch_time", {})
    if st:
        nums.add("RecSwitchMedianFrames", fmt(st["median_abs_error_frames"], 0))
        nums.add("RecSwitchWithinFive", fmt(100 * st["within_5_frames"], 0))
        nums.add("RecSwitchMissing", fmt(st["n_missing"], 0))
        nums.add("RecSwitchDecoded", fmt(st["n_decoded"], 0))
    nums.add("RecCalibrationGap", fmt(s["calibration_weighted_gap"], 3))
    nums.add("RecCalibrationMaxGap", fmt(s["calibration_max_gap"], 2))
    real = s.get("realistic", {})
    if real:
        rows = {(r["prefix"], r["length_tag"]): r for r in real["classification_by_prefix_and_length"]}
        if ("realistic_single", 0) in rows:
            nums.add("RecRealClassFull", fmt(rows[("realistic_single", 0)]["accuracy"], 2))
            nums.add("RecRealClassN", fmt(rows[("realistic_single", 0)]["n"], 0))
        short = [r["accuracy"] for (p, L), r in rows.items() if p == "realistic_single" and L in (30, 60)]
        if short:
            nums.add("RecRealClassShortMin", fmt(min(short), 2))
            nums.add("RecRealClassShortMax", fmt(max(short), 2))
        if ("realistic_still", 0) in rows:
            nums.add("RecRealStillClassFull", fmt(rows[("realistic_still", 0)]["accuracy"], 2))
        pooled = {r["prefix"]: r for r in real.get("pooled", [])}
        if "realistic_single" in pooled:
            nums.add("RecRealPooledFrameAcc", fmt(pooled["realistic_single"]["frame_accuracy"], 2))
            nums.add("RecRealPooledGain", fmt(pooled["realistic_single"]["gain"], 3))
        sd = real.get("switching_detection", {})
        dets = [v["detected_fraction"] for v in sd.get("by_prefix", {}).values()]
        if dets:
            nums.add("RecRealDetectMax", fmt(max(dets), 2))
    pooled = s.get("pooled_posterior_accuracy", [])
    if pooled:
        nums.add("RecPooledAccMin", fmt(min(r["posterior_accuracy"] for r in pooled), 2))
    pp = s.get("pooled_parameters", [])
    if pp:
        nums.add("RecPooledGainMin", fmt(min(r["gain"] for r in pp), 2))
        nums.add("RecPooledGainMax", fmt(max(r["gain"] for r in pp), 2))
        nums.add("RecPooledTauMin", fmt(min(r["tau"] for r in pp), 2))
        nums.add("RecPooledTauMax", fmt(max(r["tau"] for r in pp), 2))
        nums.add("RecPooledNavMin", fmt(min(r["N"] for r in pp), 2))
        nums.add("RecPooledNavMax", fmt(max(r["N"] for r in pp), 2))
    nums.add("RecLengths", ", ".join(str(x) for x in s["settings"]["lengths_frames"]))


def _audit_numbers(nums: Numbers) -> None:
    a = _json(RESULTS / "audit" / "michaiel2020.json")
    if a:
        nums.add("MichClips", fmt(a["n_clips"], 0))
        nums.add("MichClipsUsed", fmt(a["n_clips_used"], 0))
        ys = a.get("yaw_screen", {})
        nums.add("MichClipsExcluded", fmt(len(ys.get("excluded_clips", [])), 0))
        nums.add("MichFramesExcluded", fmt(ys.get("n_excluded_frames"), 0, True))
        rates = [e["flip_rate"] for e in ys.get("excluded_clips", [])]
        if rates:
            nums.add("MichFlipExclMin", fmt(100 * min(rates), 0))
            nums.add("MichFlipExclMax", fmt(100 * max(rates), 0))
        if "kept_flip_rate_max" in ys:
            nums.add("MichFlipKeptMax", fmt(100 * ys["kept_flip_rate_max"], 1))
        nums.add("MichAllChases", fmt(a["n_chases"], 0))
        nums.add("MichChasesMoving", fmt(a["n_chases_target_moving_gt_3cm_s"], 0))
        nums.add("MichAnimals", fmt(a["n_animals"], 0))
        nums.add("MichPxPerCm", fmt(a["px_per_cm"], 0))
        af = a.get("arena_fit_scale_px_per_cm")
        if af:
            nums.add("MichArenaScaleX", fmt(af[0], 1))
            nums.add("MichArenaScaleY", fmt(af[1], 1))
        nums.add("MichCricketTrackedPct", fmt(100 * a["cricket_tracked_fraction_mean"], 0))
    h = _json(RESULTS / "audit" / "holmgren2021.json")
    if h:
        nums.add("HolmAllChases", fmt(h.get("n_chases_shared_rule", h.get("n_chases")), 0))
        nums.add("HolmAnimals", fmt(h["n_animals"], 0))
    m = _json(RESULTS / "audit" / "mabe22.json")
    if m:
        nums.add("MabeAllChases", fmt(m["n_chases"], 0, True))
        nums.add("MabeSequences", fmt(m.get("n_sequences", m.get("n_clips")), 0, True))
        nums.add("MabeLabelledBouts", fmt(m.get("n_labelled_chase_bouts", m.get("n_labelled_bouts")), 0))


def _loao_numbers(nums: Numbers) -> None:
    t = _csv(FITS / "holmgren2021" / "leave_one_animal_out.csv")
    if t.empty:
        nums.missing.append("leave-one-animal-out")
        return
    for frames, tag in (("all", ""), ("informative", "Inf")):
        sub = t[t["frames"] == frames].set_index("model")
        for m, name in (
            ("switching:plain", "Switching"),
            ("switching:input", "Input"),
            ("switching_ar", "AR"),
            ("null", "Null"),
        ):
            if m in sub.index:
                nums.add(f"HolmLoao{tag}Gap{name}", _signed(sub.loc[m, "diff_vs_best_single_per_frame"], 4))
                nums.add(f"HolmLoao{tag}Gap{name}Lo", _signed(sub.loc[m, "diff_lo"], 4))
                nums.add(f"HolmLoao{tag}Gap{name}Hi", _signed(sub.loc[m, "diff_hi"], 4))
        if "null" in sub.index:
            nums.add(f"HolmLoao{tag}Chases", fmt(sub.loc["null", "n_chases"], 0))


def _profile_numbers(nums: Numbers) -> None:
    """Within-state regressions, lead time by bin, and the optimal-play profiles."""
    for ds, pre in PREFIX.items():
        sr = _csv(FITS / ds / "state_regression.csv")
        for name, tag in (("pure", "Pure"), ("pn", "Pn")):
            r = sr[sr["state"] == name] if not sr.empty else sr
            if len(r):
                nums.add(f"{pre}StateSlope{tag}", fmt(r["slope"].iloc[0], 2))
                nums.add(f"{pre}StateCorr{tag}", fmt(r["corr"].iloc[0], 2))
        pb = _csv(FITS / ds / "params_by_bin.csv")
        if not pb.empty:
            nums.add(f"{pre}TauBinMin", fmt(pb["tau"].min(), 2))
            nums.add(f"{pre}TauBinMax", fmt(pb["tau"].max(), 2))
        prof = _csv(EVAL / ds / "optimal_profile.csv")
        for var, tag in (("range", "Range"), ("phase", "Phase")):
            sub = prof[prof["variable"] == var].sort_values("bin") if not prof.empty else prof
            if len(sub):
                nums.add(f"{pre}Opt{tag}First", fmt(sub["gap_mean_rad"].iloc[0] * RAD2DEG, 0))
                nums.add(f"{pre}Opt{tag}Last", fmt(sub["gap_mean_rad"].iloc[-1] * RAD2DEG, 0))


def _trigger_numbers(nums: Numbers) -> None:
    """Input-dependent against fixed transitions, and the manoeuvre-aligned steering probability
    against its chase-phase prediction (table 9)."""
    t = _csv(TABLES / "table9_switch_predictors.csv")
    if t.empty:
        nums.missing.append("table 9")
        return
    names = {
        "mouse and cricket (primary)": "Mich",
        "mouse and cricket (replication)": "Holm",
        "mouse and mouse": "Mabe",
    }
    for _, r in t.iterrows():
        pre = names.get(r["dataset"])
        if pre is None:
            continue
        nums.add(f"{pre}InputGain", _signed(r["input_minus_plain"], 4))
        nums.add(f"{pre}InputGainLo", _signed(r["input_minus_plain_lo"], 4))
        nums.add(f"{pre}InputGainHi", _signed(r["input_minus_plain_hi"], 4))
        nums.add(f"{pre}InputSharePct", fmt(100 * r["share_of_switching_gap"], 0))
        if "steer_before" in t.columns and pd.notna(r.get("steer_before")):
            nums.add(f"{pre}EventPre", fmt(r["steer_before"], 2))
            nums.add(f"{pre}EventPost", fmt(r["steer_after"], 2))
            nums.add(f"{pre}EventPhasePre", fmt(r["phase_only_before"], 2))
            nums.add(f"{pre}EventPhasePost", fmt(r["phase_only_after"], 2))
            if r["steer_change"]:
                nums.add(f"{pre}EventPhaseSharePct", fmt(100 * r["phase_only_change"] / r["steer_change"], 0))


def collect() -> Numbers:
    nums = Numbers()
    for ds, pre in PREFIX.items():
        _fit_numbers(nums, _json(FITS / ds / "fit_report.json"), pre)
        _eval_numbers(nums, _json(EVAL / ds / "evaluate_report.json"), pre)
        _fit_numbers(nums, _json(FITS_FOLLOWUP / ds / "fit_report.json"), pre + "Four")
    _fit_numbers(nums, _json(FITS / "michaiel2020_arena_scale" / "fit_report.json"), "MichScale")
    _loao_numbers(nums)
    _profile_numbers(nums)
    _trigger_numbers(nums)
    _recovery_numbers(nums, _json(RESULTS / "recovery" / "summary.json"))
    _audit_numbers(nums)
    return nums


def readme_phrases(nums: Numbers | None = None) -> list[str]:
    """Phrases the README must contain, built from the same numbers as the paper."""
    m = (nums or collect()).macros
    return [
        f"a delay of {m['MichDelayMs']} to {m['MabeDelayMs']} ms",
        f"a time constant of {m['MabeTimeConstant']} to {m['MichTimeConstant']} s",
        f"holds {m['HolmSteerPathPct']} to {m['MichSteerPathPct']} percent",
    ]


def write_numbers(path: Path = PAPER / "numbers.tex") -> Numbers:
    nums = collect()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nums.tex(), encoding="utf-8")
    return nums


# ----------------------------------------------------------------------------- tables


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


HEADER_WORDS = {
    "diff_vs_best_single": "diff",
    "tau": "lead time",
    "sigma": "noise",
    "rho": "residual AR",
    "diff_vs_best_single_per_frame": "diff",
    "input_minus_plain": "input gain",
    "input_minus_plain_lo": "lo",
    "input_minus_plain_hi": "hi",
    "share_of_switching_gap": "share",
    "event_chases": "chases",
    "steer_before": "before",
    "steer_after": "after",
    "phase_only_before": "phase before",
    "phase_only_after": "phase after",
    "steer_change": "change",
    "phase_only_change": "phase change",
    "diff_informative": "diff, moving",
    "diff_informative_lo": "lo",
    "diff_informative_hi": "hi",
    "diff_lo": "lo",
    "diff_hi": "hi",
    "per_frame": "per frame",
    "delay_frames": "delay",
    "delay_ms": "delay ms",
    "gain_lo": "lo",
    "gain_hi": "hi",
    "tau_lo": "lo",
    "tau_hi": "hi",
    "N_lo": "lo",
    "N_hi": "hi",
    "sigma_lo": "lo",
    "sigma_hi": "hi",
    "outlier_fraction": "outlier",
    "stay_probability_mean": "stay",
    "switching_gap_observed": "gap",
    "switching_gap_null_q95": "null q95",
    "switching_p_value": "null p",
    "switching_gap": "gap",
    "switching_gap_lo": "lo",
    "switching_gap_hi": "hi",
    "comparator_gap": "comparator",
    "p_steering_all_frames": "p steer",
    "p_steering_target_moving": "p steer, moving",
    "frac_steering_path": "steer path",
    "frac_persistence": "persist",
    "switches_per_chase": "switches",
    "chases_with_switch": "with switch",
    "state_roles": "roles",
    "distance_from_optimal_deg_median": "dist opt (deg)",
    "distance_q25_deg": "q25",
    "distance_q75_deg": "q75",
    "distance_target_moving_deg": "dist, moving",
    "distance_to_pure_pursuit_deg": "dist pure",
    "contact_chases": "contact",
    "capture_time_ratio_median": "capture ratio",
    "manoeuvre_coef_steering": "man. coef",
    "manoeuvre_coef_lo": "lo",
    "manoeuvre_coef_hi": "hi",
    "manoeuvre_null_mean": "null mean",
    "manoeuvre_within_chase_effect": "within",
    "manoeuvre_permutation_p": "perm. p",
    "n_manoeuvres": "onsets",
    "cross_correlation_zero_lag": "xcorr 0",
    "cross_correlation_peak": "xcorr peak",
    "cross_correlation_peak_lag_frames": "peak lag",
    "held_informative_frames": "held moving",
    "informative_frames": "moving frames",
    "median_duration_s": "median s",
    "n_chases": "chases",
    "n_frames": "frames",
    "dev_chases": "dev chases",
    "held_chases": "held chases",
    "null_p": "null p",
    "p_steering": "p steer",
}


def _header(c: str) -> str:
    return _tex_escape(HEADER_WORDS.get(c, c.replace("_", " ")))


def df_to_tex(df: pd.DataFrame, caption: str, label: str, nd: int = 3, small: bool = True) -> str:
    """A booktabs table; wide tables (more than seven columns) are scaled to the text width."""
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    wide = len(cols) > 7
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\footnotesize" if small else ""),
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        (r"\resizebox{\textwidth}{!}{%" if wide else ""),
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_header(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(_cell(v, c, nd))
            elif isinstance(v, (int,)) or (hasattr(v, "dtype") and "int" in str(getattr(v, "dtype", ""))):
                cells.append(f"{int(v):,}")
            else:
                cells.append(
                    _tex_escape(
                        str(v)
                        .replace("_", " ")
                        .replace("single:", "single ")
                        .replace("switching:", "switching ")
                    )
                )
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ("}" if wide else ""), r"\end{table}", ""]
    return "\n".join(lines)


def _num(v, nd: int) -> str:
    """Fixed decimals for a table cell; a value that rounds to zero prints as unsigned zero."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    out = f"{float(v):.{nd}f}"
    return f"{0:.{nd}f}" if float(out) == 0 else out


def _cell(v, c: str, nd: int) -> str:
    """Format one numeric cell by its column: angles and milliseconds to one decimal, large
    values as integers with a thousands separator."""
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    if isinstance(v, float) and abs(v) >= 1e4:
        return f"{v:,.0f}"
    if c.endswith("_frames") and float(v).is_integer():
        return f"{int(v):,}"
    return _num(v, 1 if ("deg" in c or c.endswith("_ms")) else nd)


SHORT_DATASET = {
    "mouse and cricket (primary)": "cricket, primary",
    "mouse and cricket (replication)": "cricket, replication",
    "mouse and mouse": "mouse",
}
SHORT_LIBRARY = {"three laws": "3 laws", "three laws and persistence": "+ persistence"}
ROLE_CODES = {"steering": "S", "non-steering": "N"}
ROLE_NOTE = " Roles: S steering, N non-steering, for the states pure pursuit, lead pursuit and proportional navigation, then persistence where present."


def _pvalue(p) -> str:
    """A permutation or bootstrap p-value; values below 0.005 print as below it."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "$<$0.005" if float(p) < 0.005 else _num(p, 3)


def _interval(lo, hi, nd: int) -> str:
    return f"{_num(lo, nd)} to {_num(hi, nd)}"


def _row_id(r) -> str:
    if "dataset" not in r.index:
        return _tex_escape(str(r.iloc[0]))
    name = SHORT_DATASET.get(r["dataset"], str(r["dataset"]))
    if "library" in r.index:
        name += ", " + SHORT_LIBRARY.get(r["library"], str(r["library"]))
    return name


def _stack(text: str) -> str:
    """A header cell over one or two lines."""
    parts = text.split("|")
    if len(parts) == 1:
        return text
    return "\\shortstack[c]{" + "\\\\".join(parts) + "}"


COMPACT = {
    "table3_parameters": [
        ("dataset", _row_id),
        ("delay|(ms)", lambda r: _num(r["delay_ms"], 1)),
        ("gain|per frame", lambda r: _num(r["gain"], 3)),
        ("gain|interval", lambda r: _interval(r["gain_lo"], r["gain_hi"], 3)),
        ("lead time|(s)", lambda r: _num(r["tau"], 3)),
        ("navigation|constant", lambda r: _num(r["N"], 3)),
        ("switching|gap", lambda r: _num(r["switching_gap_observed"], 4)),
        ("null|p", lambda r: _pvalue(r["switching_p_value"])),
    ],
    "table4_posteriors": [
        ("dataset", _row_id),
        ("steering|probability", lambda r: _num(r["p_steering_all_frames"], 2)),
        ("steering,|target moving", lambda r: _num(r["p_steering_target_moving"], 2)),
        ("steering share|of frames", lambda r: _num(r["frac_steering_path"], 2)),
        ("switches|per chase", lambda r: _num(r["switches_per_chase"], 2)),
        ("chases with|a switch", lambda r: _num(r["chases_with_switch"], 2)),
    ],
    "table5_game": [
        ("dataset", _row_id),
        ("from optimal|(deg)", lambda r: _num(r["distance_from_optimal_deg_median"], 1)),
        ("from pure|pursuit (deg)", lambda r: _num(r["distance_to_pure_pursuit_deg"], 1)),
        ("capture|time ratio", lambda r: _num(r["capture_time_ratio_median"], 2)),
        ("within-chase|effect", lambda r: _num(r["manoeuvre_within_chase_effect"], 2)),
        ("permutation|p", lambda r: _pvalue(r["manoeuvre_permutation_p"])),
        ("manoeuvre|onsets", lambda r: f"{int(r['n_manoeuvres']):,}"),
    ],
    "table6_followup": [
        ("dataset, library", _row_id),
        ("navigation|constant", lambda r: _num(r["N"], 3)),
        ("navigation constant|interval", lambda r: _interval(r["N_lo"], r["N_hi"], 3)),
        ("persistence|share", lambda r: _num(r["frac_persistence"], 2)),
        ("switching|gap", lambda r: _num(r["switching_gap"], 4)),
        ("null|p", lambda r: _pvalue(r["null_p"])),
    ],
}
SPLIT_ONLY = {"table8_scale_sensitivity"}  # supplementary tables that are split rather than scaled
SPLIT_MAX_COLS = 5


def compact_tex(df: pd.DataFrame, columns: list, caption: str, label: str) -> str:
    """A main-text table with the key columns only, at small (not reduced) font size."""
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_stack(h) for h, _ in columns) + r" \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(" & ".join(fn(r) for _, fn in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _units(cols: list[str]) -> list[list[str]]:
    """Columns grouped so that an estimate stays next to its interval or quartiles."""
    partners = {
        "manoeuvre_coef_steering": ["manoeuvre_coef_lo", "manoeuvre_coef_hi"],
        "distance_from_optimal_deg_median": ["distance_q25_deg", "distance_q75_deg"],
    }
    used, units = set(), []
    for c in cols:
        if c in used:
            continue
        group = [c] + [x for x in partners.get(c, [f"{c}_lo", f"{c}_hi"]) if x in cols]
        used.update(group)
        units.append(group)
    return units


def split_tex(df: pd.DataFrame, caption: str, label: str, nd: int) -> str:
    """A full table at normal font size, its columns split into parts that fit the page."""
    ids = [c for c in ("dataset", "library", "scale") if c in df.columns]
    id_header = "dataset" if "dataset" in ids else ids[0]
    values = [c for c in df.columns if c not in ids]
    chunks, cur = [], []
    for unit in _units(values):
        if cur and len(cur) + len(unit) > SPLIT_MAX_COLS:
            chunks.append(cur)
            cur = []
        cur = cur + unit
    if cur:
        chunks.append(cur)
    note = ROLE_NOTE if "state_roles" in df.columns else ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}{note}}}",
        f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
    ]
    for k, chunk in enumerate(chunks):
        align = "l" + "r" * len(chunk)
        lines += [
            (r"\\[6pt]" if k else ""),
            f"\\begin{{tabular}}{{{align}}}",
            r"\toprule",
            " & ".join([id_header] + [_header(c) for c in chunk]) + r" \\",
            r"\midrule",
        ]
        for _, r in df.iterrows():
            cells = [_row_id(r)]
            for c in chunk:
                v = r[c]
                if c == "state_roles":
                    cells.append(" ".join(ROLE_CODES.get(w, w) for w in str(v).split()))
                elif c.endswith("_p") or c.endswith("p_value") or c == "null_p":
                    cells.append(_pvalue(v))
                elif isinstance(v, float):
                    cells.append(_cell(v, c, nd))
                elif (hasattr(v, "dtype") and "int" in str(v.dtype)) or isinstance(v, int):
                    cells.append(f"{int(v):,}")
                else:
                    cells.append(_tex_escape(str(v)))
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


TABLE_SPECS = {
    "table1_datasets": (
        "Datasets and splits. Informative frames are those with the target faster than 3 cm/s. For the mouse-and-mouse data, mouse identities are not linked across one-minute sequences, so the animal count is the count of mice per sequence.",
        "tab:datasets",
        2,
    ),
    "table2_model_comparison": (
        "Held-out log-likelihood per frame of each model minus the best single law, with chase-level bootstrap intervals, on all frames and on informative frames.",
        "tab:comparison",
        4,
    ),
    "table3_parameters": (
        "Fitted parameters of the switching model on the development sets with chase-level bootstrap intervals, and the switching gap against its parametric-bootstrap null.",
        "tab:parameters",
        3,
    ),
    "table4_posteriors": (
        "Posterior summaries on the held-out chases: steering probability, decoded state shares and switches.",
        "tab:posteriors",
        3,
    ),
    "table5_game": ("Distance from optimal play and the two-player test.", "tab:game", 3),
    "table6_followup": (
        "The post-freeze follow-up with a persistence state, against the frozen three-law analysis.",
        "tab:followup",
        3,
    ),
    "table7_leave_one_animal_out": (
        "Replication set scored leave-one-animal-out: held-out log-likelihood per frame minus the best single law, pooled over three folds, with chase-level bootstrap intervals.",
        "tab:loao",
        4,
    ),
    "table9_switch_predictors": (
        "What predicts switching. Held-out log-likelihood per frame of input-dependent minus fixed transitions, with chase-level bootstrap intervals, and its share of the switching gap; steering probability from 0.5 to 0.1 s before and from 0.1 to 0.5 s after a target manoeuvre, with the prediction from chase phase alone.",
        "tab:inputs",
        4,
    ),
    "table8_scale_sensitivity": (
        "Primary dataset at the deposit scale and at the arena-fit scale.",
        "tab:scale",
        3,
    ),
}


def write_tables(out_dir: Path = PAPER / "tables") -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (caption, label, nd) in TABLE_SPECS.items():
        df = _csv(TABLES / f"{name}.csv")
        if df.empty:
            continue
        if name in COMPACT:
            main_caption = (
                f"{caption} Key columns; all columns are in Table~\\ref{{{label}full}} in the supplement."
            )
            (out_dir / f"{name}.tex").write_text(
                compact_tex(df, COMPACT[name], main_caption, label), encoding="utf-8"
            )
            full_caption = f"All columns of Table~\\ref{{{label}}}. {caption}"
            (out_dir / f"{name}_full.tex").write_text(
                split_tex(df, full_caption, f"{label}full", nd), encoding="utf-8"
            )
            written += [name, f"{name}_full"]
        elif name in SPLIT_ONLY:
            (out_dir / f"{name}.tex").write_text(split_tex(df, caption, label, nd), encoding="utf-8")
            written.append(name)
        else:
            (out_dir / f"{name}.tex").write_text(df_to_tex(df, caption, label, nd), encoding="utf-8")
            written.append(name)
    return written


# ----------------------------------------------------------------------------- build


def build(pdf: bool = True) -> Path | None:
    """Regenerate the numbers and tables, copy the figures, and compile with latexmk. Needs the
    LaTeX source of the paper, which is not part of the published repository."""
    if not (PAPER / "main.tex").exists():
        raise SystemExit(
            "The LaTeX source of the paper is not part of the published repository; "
            "the built paper is paper/main.pdf."
        )
    write_numbers()
    write_tables()
    fig_dir = PAPER / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted((RESULTS / "figures").glob("fig*.pdf")):
        shutil.copy2(p, fig_dir / p.name)
    if not pdf:
        return None
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk not found; install a TeX distribution or run with pdf=False")
    subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-silent", "main.tex"],
        cwd=PAPER,
        check=True,
    )
    return PAPER / "main.pdf"
