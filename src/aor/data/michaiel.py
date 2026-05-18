"""Loader for the Michaiel, Abe and Niell prey-capture behavioural data (Dryad 10.5061/dryad.8cz8w9gmw).

The deposited MATLAB file holds 105 clips from 7 mice hunting live crickets. Per clip it gives the
mouse head centre and the cricket centre in pixels of the overhead camera, the head yaw measured
from the DeepLabCut head points, and the authors' approach flags. The overhead camera ran at 30
frames per second; the deposited series were resampled onto a 60 Hz grid shared with the eye
cameras, so this loader keeps every second sample and works at 30 Hz. The authors' calibration is
27 pixels per centimetre.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from ..contracts import Chase, ChaseMeta, ContractError, build_frames, validate_chase, wrap_angle
from ..kinematics import (
    angular_difference,
    despike,
    heading_from_velocity,
    interpolate_gaps,
    interpolate_gaps_xy,
    repair_flips,
    velocity,
)
from ..kinematics import (
    speed as speed_of,
)
from ..segment import SegmentationParams, segment_chases

DATASET = "michaiel2020"
FILE = "Michaiel_et_al.2020_fullDataset.mat"
STORED_FPS = 60.0
FPS = 30.0
PX_PER_CM = 27.0
ARENA_CM = (45.0, 38.0)  # the arena size stated by the authors, used by the arena-fit scale
MAX_MOUSE_SPEED = 150.0  # cm/s, despike threshold for the head centre
MAX_CRICKET_SPEED = 300.0  # cm/s, despike threshold for the cricket
GAP_FRAMES = 5  # interpolate position gaps up to this many frames at 30 Hz
YAW_FLIP_THRESHOLD = np.pi / 2  # a yaw sample this far from both neighbours, which agree, is a flip
FLIP_RAD = (
    1.0  # a yaw change above this between consecutive frames (1700 degrees per second) is a tracking flip
)
MAX_FLIP_RATE = (
    0.01  # clips with more flips than this fraction of frame pairs have an untracked yaw and are excluded
)


@dataclass
class Trial:
    index: int
    animal: str
    session: str
    clip: str
    mouse_xy: np.ndarray  # (T, 2) cm, 30 Hz
    cricket_xy: np.ndarray  # (T, 2) cm, 30 Hz
    head_theta_deg: np.ndarray  # (T,) authors' head yaw in degrees
    approach_flag: np.ndarray  # (T,) authors' approach epochs


def _cell(m: dict, key: str) -> list:
    return list(np.asarray(m[key], dtype=object).ravel())


def read_trials(raw_dir: Path) -> list[Trial]:
    m = sio.loadmat(Path(raw_dir) / FILE, squeeze_me=True, struct_as_record=False)
    names = [str(x) for x in _cell(m, "animalName")]
    sess = [str(x) for x in _cell(m, "expSession")]
    clips = [str(x) for x in _cell(m, "clipNumber")]
    mx, cx, ht, ap = (_cell(m, k) for k in ("mouse_xy", "cricket_xy", "headTheta", "approachEpochs"))
    trials = []
    for i in range(len(names)):
        mouse = np.atleast_2d(np.asarray(mx[i], dtype=float)).T[::2] / PX_PER_CM
        cricket = np.atleast_2d(np.asarray(cx[i], dtype=float)).T[::2] / PX_PER_CM
        theta = np.asarray(ht[i], dtype=float).ravel()[::2]
        flag = np.nan_to_num(np.asarray(ap[i], dtype=float).ravel())[::2] > 0.5
        n = min(len(mouse), len(cricket), len(theta))
        flag = np.pad(flag[:n], (0, max(0, n - len(flag))))
        trials.append(Trial(i, names[i], sess[i], clips[i], mouse[:n], cricket[:n], theta[:n], flag[:n]))
    return trials


def head_theta_to_heading(theta_deg: np.ndarray, sign: float, offset: float) -> np.ndarray:
    """Convert the authors' head yaw (degrees) to a heading in the arena frame used here."""
    return wrap_angle(sign * np.deg2rad(theta_deg) + offset)


def calibrate_heading_convention(trials: list[Trial]) -> tuple[float, float, float]:
    """Pick the sign and offset that best align the authors' head yaw with the direction of
    travel when the mouse runs fast. Returns (sign, offset, median absolute error in degrees)."""
    best = None
    for sign in (1.0, -1.0):
        for offset in (0.0, np.pi / 2, np.pi, -np.pi / 2):
            errs = []
            for tr in trials:
                xy = interpolate_gaps_xy(despike(tr.mouse_xy, FPS, MAX_MOUSE_SPEED), GAP_FRAMES)
                hv = heading_from_velocity(xy, FPS, window=5, min_speed=15.0)
                h = head_theta_to_heading(tr.head_theta_deg, sign, offset)
                ok = np.isfinite(hv) & np.isfinite(h)
                if ok.sum():
                    errs.append(np.abs(angular_difference(h[ok], hv[ok])))
            err = np.degrees(np.median(np.concatenate(errs))) if errs else np.inf
            if best is None or err < best[2]:
                best = (sign, offset, err)
    return best


def arena_fit_scale(trials: list[Trial]) -> tuple[float, float]:
    """Pixels per centimetre along x and y such that the extent of the tracked head positions
    (0.5th to 99.5th percentile over all clips) matches the stated arena size."""
    xy = np.concatenate([t.mouse_xy for t in trials]) * PX_PER_CM
    xy = xy[np.isfinite(xy).all(axis=1)]
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    return float((hi[0] - lo[0]) / ARENA_CM[0]), float((hi[1] - lo[1]) / ARENA_CM[1])


def trial_to_chases(
    tr: Trial,
    params: SegmentationParams,
    heading_sign: float,
    heading_offset: float,
    scale: tuple[float, float] = (PX_PER_CM, PX_PER_CM),
) -> tuple[list[Chase], list[str]]:
    """Segment one clip into chases and build contract tables. Returns (chases, rejections).
    ``scale`` gives pixels per centimetre along x and y; an anisotropic scale rescales the
    positions and corrects the head yaw, which was measured in the pixel frame."""
    factor = np.array([PX_PER_CM / scale[0], PX_PER_CM / scale[1]])
    mouse = interpolate_gaps_xy(despike(tr.mouse_xy * factor, FPS, MAX_MOUSE_SPEED), GAP_FRAMES)
    cricket = interpolate_gaps_xy(despike(tr.cricket_xy * factor, FPS, MAX_CRICKET_SPEED), GAP_FRAMES)
    heading = head_theta_to_heading(tr.head_theta_deg, heading_sign, heading_offset)
    if scale[0] != scale[1]:
        heading = np.arctan2(np.sin(heading) * factor[1], np.cos(heading) * factor[0])
    heading = repair_flips(heading, YAW_FLIP_THRESHOLD)  # repairs single-frame yaw flips only
    # interpolate short heading gaps on the unit circle
    ux = interpolate_gaps(np.cos(heading), GAP_FRAMES)
    uy = interpolate_gaps(np.sin(heading), GAP_FRAMES)
    heading = wrap_angle(np.arctan2(uy, ux))
    p_speed = speed_of(mouse, FPS, window=5)
    t_vel = velocity(cricket, FPS, window=5)
    segs = segment_chases(mouse[:, 0], mouse[:, 1], heading, cricket[:, 0], cricket[:, 1], FPS, params)
    chases, rejections = [], []
    for k, g in enumerate(segs):
        sl = slice(g.start, g.end)
        frame = np.arange(g.start, g.end)
        df = build_frames(
            frame,
            FPS,
            mouse[sl, 0],
            mouse[sl, 1],
            heading[sl],
            p_speed[sl],
            cricket[sl, 0],
            cricket[sl, 1],
            t_vel[sl, 0],
            t_vel[sl, 1],
            px_pix=tr.mouse_xy[sl, 0] * PX_PER_CM,
            py_pix=tr.mouse_xy[sl, 1] * PX_PER_CM,
            tx_pix=tr.cricket_xy[sl, 0] * PX_PER_CM,
            ty_pix=tr.cricket_xy[sl, 1] * PX_PER_CM,
            t_speed=np.hypot(t_vel[sl, 0], t_vel[sl, 1]),
        )
        meta = ChaseMeta(
            dataset=DATASET,
            session_id=f"{tr.animal}_t{tr.index:03d}",
            animal_id=tr.animal,
            chase_id=f"{tr.animal}_t{tr.index:03d}_{k:02d}",
            fps=FPS,
            start_frame=int(g.start),
            end_frame=int(g.end - 1),
            end_reason=g.end_reason,
            target_type="cricket",
            pursuer_point="head_centre",
            px_per_cm=float(scale[0]),
            native_fps=30.0,
            coordinate_frame="overhead_camera_px",
            extra={
                "trial_index": tr.index,
                "px_per_cm_y": float(scale[1]),
                "session": tr.session,
                "clip": tr.clip,
                "authors_approach_fraction": float(tr.approach_flag[sl].mean()),
            },
        )
        chase = Chase(meta=meta, frames=df)
        try:
            validate_chase(chase, min_duration_s=params.min_duration_s)
        except ContractError as e:
            rejections.append(str(e).splitlines()[1].strip() if "\n" in str(e) else str(e))
            continue
        chases.append(chase)
    return chases, rejections


def yaw_flip_rate(theta_deg: np.ndarray) -> float:
    """Fraction of consecutive finite yaw samples that differ by more than FLIP_RAD."""
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    d = np.angle(np.exp(1j * np.diff(th)))
    d = d[np.isfinite(d)]
    return float(np.mean(np.abs(d) > FLIP_RAD)) if len(d) else float("nan")


def yaw_screen(trials: list[Trial]) -> tuple[list[Trial], list[dict]]:
    """Keep the clips whose yaw track passes the flip-rate screen; list the excluded ones."""
    kept, excluded = [], []
    for tr in trials:
        rate = yaw_flip_rate(tr.head_theta_deg)
        if rate > MAX_FLIP_RATE:
            excluded.append(
                {
                    "animal": tr.animal,
                    "session": tr.session,
                    "clip": tr.clip,
                    "flip_rate": round(rate, 4),
                    "n_frames": len(tr.head_theta_deg),
                }
            )
        else:
            kept.append(tr)
    return kept, excluded


def load_chases(
    raw_dir: Path, params: SegmentationParams = SegmentationParams(), scale: str | None = None
) -> tuple[list[Chase], dict]:
    """All chases of the deposit. ``scale`` None uses the 27 px/cm of the deposit; "arena_scale"
    uses the anisotropic arena-fit scale (the sensitivity analysis)."""
    all_trials = read_trials(Path(raw_dir))
    trials, excluded = yaw_screen(all_trials)
    sign, offset, err = calibrate_heading_convention(trials)
    if scale is None:
        sc = (PX_PER_CM, PX_PER_CM)
    elif scale == "arena_scale":
        sc = arena_fit_scale(all_trials)
    else:
        raise ValueError(f"unknown scale {scale!r}")
    chases, rejections = [], []
    for tr in trials:
        c, r = trial_to_chases(tr, params, sign, offset, sc)
        chases.extend(c)
        rejections.extend(r)
    info = {
        "heading_sign": sign,
        "heading_offset_rad": offset,
        "heading_vs_travel_median_abs_error_deg": err,
        "scale_px_per_cm": list(sc),
        "arena_fit_scale_px_per_cm": list(arena_fit_scale(all_trials)),
        "n_trials": len(all_trials),
        "n_trials_used": len(trials),
        "yaw_screen": {
            "max_flip_rate": MAX_FLIP_RATE,
            "flip_rad": FLIP_RAD,
            "excluded_clips": excluded,
            "n_excluded_frames": int(sum(e["n_frames"] for e in excluded)),
            "kept_flip_rate_median": float(np.nanmedian([yaw_flip_rate(t.head_theta_deg) for t in trials])),
            "kept_flip_rate_max": float(np.nanmax([yaw_flip_rate(t.head_theta_deg) for t in trials])),
        },
        "n_rejected": len(rejections),
        "rejection_reasons": rejections,
    }
    return chases, info


def audit(raw_dir: Path, params: SegmentationParams = SegmentationParams()) -> dict:
    """Facts about the data and the provisional chase counts, without any model fitting."""
    trials = read_trials(Path(raw_dir))
    chases, info = load_chases(raw_dir, params)
    animals = sorted({t.animal for t in trials})
    minutes = {a: sum(len(t.mouse_xy) for t in trials if t.animal == a) / FPS / 60 for a in animals}
    dur = np.array([c.duration_s for c in chases])
    tspeed = np.array([float(np.nanmean(c.frames["t_speed"])) for c in chases])
    moving = tspeed > 3.0
    per_animal = {a: int(sum(c.meta.animal_id == a for c in chases)) for a in animals}
    per_animal_moving = {
        a: int(sum((c.meta.animal_id == a) and mv for c, mv in zip(chases, moving, strict=True)))
        for a in animals
    }
    cricket_valid = float(np.mean([np.isfinite(t.cricket_xy[:, 0]).mean() for t in trials]))
    return {
        "dataset": DATASET,
        "fps_native": 30,
        "fps_stored": 60,
        "fps_analysis": FPS,
        "px_per_cm": PX_PER_CM,
        "arena_fit_scale_px_per_cm": info["arena_fit_scale_px_per_cm"],
        "n_animals": len(animals),
        "n_clips": len(trials),
        "n_clips_used": info["n_trials_used"],
        "yaw_screen": info["yaw_screen"],
        "minutes_per_animal": {k: round(v, 1) for k, v in minutes.items()},
        "cricket_tracked_fraction_mean": round(cricket_valid, 3),
        "heading": {
            k: info[k]
            for k in ("heading_sign", "heading_offset_rad", "heading_vs_travel_median_abs_error_deg")
        },
        "segmentation": params.__dict__,
        "n_chases": len(chases),
        "n_rejected_by_contract": info["n_rejected"],
        "chase_duration_s": {
            "median": float(np.median(dur)),
            "q25": float(np.percentile(dur, 25)),
            "q75": float(np.percentile(dur, 75)),
            "max": float(dur.max()),
        }
        if len(dur)
        else None,
        "n_chases_ge_1s": int((dur >= 1).sum()),
        "n_chases_ge_2s": int((dur >= 2).sum()),
        "n_chases_target_moving_gt_3cm_s": int(moving.sum()),
        "end_reasons": {
            r: int(sum(c.meta.end_reason == r for c in chases))
            for r in sorted({c.meta.end_reason for c in chases})
        },
        "chases_per_animal": per_animal,
        "chases_per_animal_target_moving": per_animal_moving,
    }
