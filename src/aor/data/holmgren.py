"""Loader for the Holmgren et al. mouse prey-pursuit tracking data (Dryad 10.5061/dryad.2z34tmpnc).

Three mice hunted crickets with olfactory and auditory cues masked. Head position and
orientation (a quaternion) and the three-dimensional cricket position were tracked at 200 Hz in
the coordinate frame of a laser scan of the arena, in centimetres. The authors also provide
frame lists for their detect, chase (tracking), and capture phases. Horizontal positions are
the x and y columns of the scan frame; the head heading is the horizontal projection of the
head's forward axis, with the axis chosen by comparison against the direction of travel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import Chase, ChaseMeta, ContractError, build_frames, validate_chase, wrap_angle
from ..kinematics import (
    angular_difference,
    heading_from_velocity,
    interpolate_gaps,
    interpolate_gaps_xy,
    velocity,
)
from ..kinematics import (
    speed as speed_of,
)
from ..segment import SegmentationParams, segment_chases

DATASET = "holmgren2021"
FPS = 200.0
MICE = {"f001": "m001_scan.csv", "f002": "m002_scan.csv", "f003": "m003_scan.csv"}
GAP_FRAMES = 20  # 0.1 s at 200 Hz
CANDIDATE_AXES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternions q (N, 4) in (w, x, y, z) order."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v
    # standard quaternion rotation q v q*
    t2 = w * x
    t3 = w * y
    t4 = w * z
    t5 = -x * x
    t6 = x * y
    t7 = x * z
    t8 = -y * y
    t9 = y * z
    t10 = -z * z
    rx = 2 * ((t8 + t10) * vx + (t6 - t4) * vy + (t3 + t7) * vz) + vx
    ry = 2 * ((t4 + t6) * vx + (t5 + t10) * vy + (t9 - t2) * vz) + vy
    rz = 2 * ((t7 - t3) * vx + (t2 + t9) * vy + (t5 + t8) * vz) + vz
    return np.column_stack([rx, ry, rz])


def read_mouse(raw_dir: Path, mouse: str) -> pd.DataFrame:
    df = pd.read_csv(Path(raw_dir) / "trajectories" / MICE[mouse], sep=r"\s+", engine="c", na_values=["NaN"])
    return df


def phase_frames(raw_dir: Path, mouse: str, variant: str, phase: str) -> np.ndarray:
    p = Path(raw_dir) / "behavioral_classifications" / mouse / "behaviour" / variant / f"{phase}.txt"
    if not p.exists():
        return np.array([], dtype=int)
    return np.loadtxt(p, dtype=int, ndmin=1)


def segments_from_frame_list(frames: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive frame numbers -> (first, last) pairs."""
    frames = np.asarray(sorted(set(frames.tolist())))
    if frames.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(frames) > 1)
    starts = np.concatenate([[frames[0]], frames[cuts + 1]])
    ends = np.concatenate([frames[cuts], [frames[-1]]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def head_heading(df: pd.DataFrame, axis_name: str) -> np.ndarray:
    q = df[["rwh", "rxh", "ryh", "rzh"]].to_numpy(dtype=float)
    fwd = quat_rotate(q, CANDIDATE_AXES[axis_name])
    return wrap_angle(np.arctan2(fwd[:, 1], fwd[:, 0]))


def calibrate_forward_axis(df: pd.DataFrame) -> tuple[str, float]:
    """Choose the head-frame axis whose horizontal projection best matches the direction of travel
    when the head moves faster than 15 cm/s. Returns (axis name, median absolute error, degrees)."""
    xy = df[["pxh", "pyh"]].to_numpy(dtype=float)
    hv = heading_from_velocity(xy, FPS, window=21, min_speed=15.0)
    best = None
    for name in CANDIDATE_AXES:
        h = head_heading(df, name)
        ok = np.isfinite(h) & np.isfinite(hv)
        if not ok.any():
            continue
        err = float(np.degrees(np.median(np.abs(angular_difference(h[ok], hv[ok])))))
        if best is None or err < best[1]:
            best = (name, err)
    return best


def mouse_to_chases(
    raw_dir: Path,
    mouse: str,
    params: SegmentationParams,
    use_author_phases: bool = True,
) -> tuple[list[Chase], dict]:
    df = read_mouse(raw_dir, mouse)
    axis, err = calibrate_forward_axis(df)
    heading = head_heading(df, axis)
    ux = interpolate_gaps(np.cos(heading), GAP_FRAMES)
    uy = interpolate_gaps(np.sin(heading), GAP_FRAMES)
    heading = wrap_angle(np.arctan2(uy, ux))
    head = interpolate_gaps_xy(df[["pxh", "pyh"]].to_numpy(dtype=float), GAP_FRAMES)
    cricket = interpolate_gaps_xy(df[["pxc", "pyc"]].to_numpy(dtype=float), GAP_FRAMES)
    frn = df["frn"].to_numpy()
    p_speed = speed_of(head, FPS, window=21)
    t_vel = velocity(cricket, FPS, window=21)
    chases, rejections = [], []
    if use_author_phases:
        # authors' chase (tracking) segments followed by their capture segments, joined when contiguous
        chase_segs = segments_from_frame_list(
            np.concatenate(
                [
                    phase_frames(raw_dir, mouse, "allgr5", "chase"),
                    phase_frames(raw_dir, mouse, "allgr5", "capture"),
                ]
            )
        )
        windows = [(int(np.searchsorted(frn, a)), int(np.searchsorted(frn, b)) + 1) for a, b in chase_segs]
        reasons = ["contact"] * len(windows)
    else:
        segs = segment_chases(head[:, 0], head[:, 1], heading, cricket[:, 0], cricket[:, 1], FPS, params)
        windows = [(g.start, g.end) for g in segs]
        reasons = [g.end_reason for g in segs]
    for k, ((a, b), reason) in enumerate(zip(windows, reasons, strict=True)):
        sl = slice(a, b)
        if b - a < 2:
            continue
        df_c = build_frames(
            np.arange(a, b),
            FPS,
            head[sl, 0],
            head[sl, 1],
            heading[sl],
            p_speed[sl],
            cricket[sl, 0],
            cricket[sl, 1],
            t_vel[sl, 0],
            t_vel[sl, 1],
            t_speed=np.hypot(t_vel[sl, 0], t_vel[sl, 1]),
        )
        meta = ChaseMeta(
            DATASET,
            mouse,
            mouse,
            f"{mouse}_{k:03d}",
            FPS,
            int(a),
            int(b - 1),
            reason,
            "cricket",
            pursuer_point="head_tracking_origin",
            px_per_cm=None,
            native_fps=FPS,
            coordinate_frame="scan_cm",
            extra={
                "source_frame_first": int(frn[a]),
                "source_frame_last": int(frn[b - 1]),
                "author_phase": use_author_phases,
            },
        )
        chase = Chase(meta=meta, frames=df_c)
        try:
            validate_chase(chase, min_duration_s=params.min_duration_s)
        except ContractError as e:
            rejections.append(str(e).splitlines()[1].strip() if "\n" in str(e) else str(e))
            continue
        chases.append(chase)
    info = {
        "forward_axis": axis,
        "heading_vs_travel_median_abs_error_deg": err,
        "n_rows": len(df),
        "tracked_fraction_head": float(np.isfinite(df["pxh"]).mean()),
        "tracked_fraction_cricket": float(np.isfinite(df["pxc"]).mean()),
        "n_rejected": len(rejections),
        "rejection_reasons": rejections,
    }
    return chases, info


def load_chases(
    raw_dir: Path, params: SegmentationParams = SegmentationParams(), use_author_phases: bool = True
) -> tuple[list[Chase], dict]:
    all_chases, infos = [], {}
    for mouse in MICE:
        c, info = mouse_to_chases(Path(raw_dir), mouse, params, use_author_phases)
        all_chases.extend(c)
        infos[mouse] = info
    return all_chases, infos


def audit(raw_dir: Path, params: SegmentationParams = SegmentationParams()) -> dict:
    authors, info_a = load_chases(raw_dir, params, use_author_phases=True)
    ours, info_o = load_chases(raw_dir, params, use_author_phases=False)

    def summary(chases):
        if not chases:
            return {"n": 0}
        dur = np.array([c.duration_s for c in chases])
        tsp = np.array([float(np.nanmean(c.frames["t_speed"])) for c in chases])
        return {
            "n": len(chases),
            "duration_s": {
                "median": float(np.median(dur)),
                "q25": float(np.percentile(dur, 25)),
                "q75": float(np.percentile(dur, 75)),
                "max": float(dur.max()),
            },
            "n_ge_1s": int((dur >= 1).sum()),
            "n_ge_2s": int((dur >= 2).sum()),
            "n_target_moving_gt_3cm_s": int((tsp > 3).sum()),
            "per_mouse": {m: int(sum(c.meta.animal_id == m for c in chases)) for m in MICE},
            "end_reasons": {
                r: int(sum(c.meta.end_reason == r for c in chases))
                for r in sorted({c.meta.end_reason for c in chases})
            },
        }

    return {
        "dataset": DATASET,
        "fps": FPS,
        "units": "cm (scan frame)",
        "n_animals": len(MICE),
        "per_mouse_tracking": {
            m: {k: v for k, v in info_a[m].items() if k != "rejection_reasons"} for m in MICE
        },
        "author_phase_chases": summary(authors),
        "kinematic_rule_chases": summary(ours),
        "segmentation": params.__dict__,
        "rejections_author": sum(info_a[m]["n_rejected"] for m in MICE),
        "rejections_rule": sum(info_o[m]["n_rejected"] for m in MICE),
    }
