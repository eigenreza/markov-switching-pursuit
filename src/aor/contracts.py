"""Data contracts for chase tables.

Every dataset loader in this package produces the same object: a ``Chase``. A chase is one
pursuit episode, held as a pandas DataFrame with one row per video frame plus a small
metadata record. Downstream code (simulator export, inference, evaluation, figures) reads
only this format, so the contract below is the single place where units, shapes, and
invariants are defined. Violations raise ``ContractError`` with every failed check listed.

Units: centimetres for positions, seconds for time, radians for angles, centimetres per
second for speeds. Angles follow the mathematical convention (counter-clockwise positive,
zero along the +x axis of the arena frame). Headings are wrapped to (-pi, pi].

Pursuer and target positions in one chase are always expressed in the same coordinate frame,
named in ``ChaseMeta.coordinate_frame``, and the analysis frame rate never exceeds the native
rate of the recording, ``ChaseMeta.native_fps``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

# Column name -> (dtype kind, description). Kind "f" is float, "i" is integer.
REQUIRED_COLUMNS: dict[str, tuple[str, str]] = {
    "frame": ("i", "source frame index, contiguous within a chase"),
    "t": ("f", "time in seconds from the first frame of the chase"),
    "px": ("f", "pursuer reference point x, cm (head centre or body centre, see meta.pursuer_point)"),
    "py": ("f", "pursuer reference point y, cm"),
    "p_heading": ("f", "pursuer heading, radians in (-pi, pi], from the body axis or the head yaw"),
    "p_speed": ("f", "pursuer speed, cm/s, smoothed"),
    "tx": ("f", "target position x, cm"),
    "ty": ("f", "target position y, cm"),
    "tvx": ("f", "target velocity x, cm/s, smoothed finite difference"),
    "tvy": ("f", "target velocity y, cm/s, smoothed finite difference"),
    "range": ("f", "pursuer-to-target distance, cm"),
    "bearing": ("f", "angle from pursuer heading to the line of sight, radians in (-pi, pi]"),
}

# Optional columns that loaders fill when the source provides them. They carry the raw
# pixel coordinates for diagnostics and are allowed to be NaN when the source is already
# in centimetres.
OPTIONAL_COLUMNS: dict[str, tuple[str, str]] = {
    "px_pix": ("f", "pursuer reference point x in source pixels"),
    "py_pix": ("f", "pursuer reference point y in source pixels"),
    "tx_pix": ("f", "target x in source pixels"),
    "ty_pix": ("f", "target y in source pixels"),
    "t_heading": ("f", "target heading, radians, when the target has a body axis (mouse targets)"),
    "t_speed": ("f", "target speed, cm/s"),
    "law_true": ("f", "simulated data only: index of the generating law at this frame"),
}

END_REASONS = ("capture", "contact", "stall", "clip_end", "track_loss")
TARGET_TYPES = ("cricket", "mouse", "virtual", "simulated")

# Physical plausibility bounds used by the validator. They are deliberately loose: the
# purpose is to catch unit mistakes and tracking failures, not to trim real behaviour.
MAX_PURSUER_SPEED_CM_S = 200.0
MAX_TARGET_SPEED_CM_S = 400.0
MAX_RANGE_CM = 200.0
MIN_RANGE_BEFORE_END_CM = 0.0


class ContractError(ValueError):
    """Raised when a chase violates the data contract. The message lists every failure."""


@dataclass
class ChaseMeta:
    """Identity and provenance of one chase."""

    dataset: str
    session_id: str
    animal_id: str
    chase_id: str
    fps: float
    start_frame: int
    end_frame: int
    end_reason: str
    target_type: str
    pursuer_point: str = "head_centre"
    px_per_cm: float | None = None
    native_fps: float | None = None
    coordinate_frame: str = "arena_cm"
    split: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chase:
    """One pursuit episode: a per-frame table and its metadata."""

    meta: ChaseMeta
    frames: pd.DataFrame

    @property
    def duration_s(self) -> float:
        return float(len(self.frames)) / float(self.meta.fps)

    def validate(self, min_duration_s: float = 0.0) -> Chase:
        validate_chase(self, min_duration_s=min_duration_s)
        return self


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to (-pi, pi]."""
    return -((-np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi)


def _check_columns(df: pd.DataFrame, problems: list[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")
        return
    for col, (kind, _) in REQUIRED_COLUMNS.items():
        if kind == "i" and not pd.api.types.is_integer_dtype(df[col]):
            problems.append(f"column {col!r} must be integer, got {df[col].dtype}")
        if kind == "f" and not pd.api.types.is_float_dtype(df[col]):
            problems.append(f"column {col!r} must be float, got {df[col].dtype}")
    unknown = [c for c in df.columns if c not in REQUIRED_COLUMNS and c not in OPTIONAL_COLUMNS]
    if unknown:
        problems.append(f"unknown columns: {unknown}")


def _check_values(chase: Chase, problems: list[str], min_duration_s: float) -> None:
    df = chase.frames
    meta = chase.meta
    n = len(df)
    if n < 2:
        problems.append(f"chase has {n} frames, at least 2 required")
        return
    if n / meta.fps < min_duration_s:
        problems.append(f"duration {n / meta.fps:.3f} s below minimum {min_duration_s} s")

    req = [c for c in REQUIRED_COLUMNS if c in df.columns]
    nan_cols = [c for c in req if df[c].isna().any()]
    if nan_cols:
        problems.append(f"NaN present in required columns: {nan_cols}")

    frames = df["frame"].to_numpy()
    if not np.all(np.diff(frames) == 1):
        problems.append("frames are not contiguous (missing frames inside the chase)")
    if frames[0] != meta.start_frame or frames[-1] != meta.end_frame:
        problems.append(
            f"frame range {frames[0]}..{frames[-1]} disagrees with meta {meta.start_frame}..{meta.end_frame}"
        )

    t = df["t"].to_numpy()
    dt = np.diff(t)
    if t[0] != 0.0:
        problems.append(f"t must start at 0, got {t[0]}")
    if np.any(dt <= 0):
        problems.append("t is not strictly increasing")
    elif not np.allclose(dt, 1.0 / meta.fps, rtol=1e-3, atol=1e-6):
        problems.append(f"frame spacing deviates from 1/fps = {1.0 / meta.fps:.6f} s")

    px, py, tx, ty = (df[c].to_numpy() for c in ("px", "py", "tx", "ty"))
    rng = df["range"].to_numpy()
    rng_check = np.hypot(px - tx, py - ty)
    if not np.allclose(rng, rng_check, atol=1e-6):
        problems.append("range column disagrees with pursuer-target distance")
    if np.any(rng[:-1] <= MIN_RANGE_BEFORE_END_CM):
        problems.append("pursuer and target coincide before the last frame")
    if np.any(rng > MAX_RANGE_CM):
        problems.append(f"range exceeds {MAX_RANGE_CM} cm, check units")

    heading = df["p_heading"].to_numpy()
    if np.any(heading <= -np.pi) or np.any(heading > np.pi):
        problems.append("p_heading outside (-pi, pi]")
    bearing = df["bearing"].to_numpy()
    if np.any(bearing <= -np.pi) or np.any(bearing > np.pi):
        problems.append("bearing outside (-pi, pi]")
    los = np.arctan2(ty - py, tx - px)
    bearing_check = wrap_angle(los - heading)
    if not np.allclose(wrap_angle(bearing - bearing_check), 0.0, atol=1e-6):
        problems.append("bearing column disagrees with line of sight minus heading")

    if np.any(df["p_speed"].to_numpy() < 0):
        problems.append("negative pursuer speed")
    if np.any(df["p_speed"].to_numpy() > MAX_PURSUER_SPEED_CM_S):
        problems.append(f"pursuer speed exceeds {MAX_PURSUER_SPEED_CM_S} cm/s, check units")
    tspeed = np.hypot(df["tvx"].to_numpy(), df["tvy"].to_numpy())
    if np.any(tspeed > MAX_TARGET_SPEED_CM_S):
        problems.append(f"target speed exceeds {MAX_TARGET_SPEED_CM_S} cm/s, check units")


def _check_meta(meta: ChaseMeta, problems: list[str]) -> None:
    if meta.fps <= 0:
        problems.append(f"fps must be positive, got {meta.fps}")
    if meta.native_fps is not None and meta.fps > meta.native_fps * (1.0 + 1e-9):
        problems.append(f"analysis rate {meta.fps} exceeds the native rate {meta.native_fps}")
    if not meta.coordinate_frame:
        problems.append("coordinate_frame is empty")
    if meta.end_reason not in END_REASONS:
        problems.append(f"end_reason {meta.end_reason!r} not in {END_REASONS}")
    if meta.target_type not in TARGET_TYPES:
        problems.append(f"target_type {meta.target_type!r} not in {TARGET_TYPES}")
    if meta.end_frame < meta.start_frame:
        problems.append("end_frame before start_frame")
    for name in ("dataset", "session_id", "animal_id", "chase_id"):
        if not getattr(meta, name):
            problems.append(f"meta.{name} is empty")


def validate_chase(chase: Chase, min_duration_s: float = 0.0) -> None:
    """Check one chase against the contract. Raises ContractError listing all failures."""
    problems: list[str] = []
    _check_meta(chase.meta, problems)
    _check_columns(chase.frames, problems)
    if not problems or all("missing required" not in p for p in problems):
        _check_values(chase, problems, min_duration_s)
    if problems:
        head = (
            f"chase {chase.meta.dataset}/{chase.meta.session_id}/{chase.meta.chase_id} violates the contract:"
        )
        raise ContractError("\n  - ".join([head, *problems]))


def validate_all(chases: Iterable[Chase], min_duration_s: float = 0.0) -> list[Chase]:
    """Validate every chase; raise on the first failure with its full message."""
    out = []
    for c in chases:
        validate_chase(c, min_duration_s=min_duration_s)
        out.append(c)
    return out


def build_frames(
    frame: np.ndarray,
    fps: float,
    px: np.ndarray,
    py: np.ndarray,
    p_heading: np.ndarray,
    p_speed: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    tvx: np.ndarray,
    tvy: np.ndarray,
    **optional: np.ndarray,
) -> pd.DataFrame:
    """Assemble a contract-conforming frame table, deriving t, range, and bearing."""
    frame = np.asarray(frame, dtype=np.int64)
    px, py, tx, ty = (np.asarray(a, dtype=float) for a in (px, py, tx, ty))
    heading = wrap_angle(np.asarray(p_heading, dtype=float))
    los = np.arctan2(ty - py, tx - px)
    df = pd.DataFrame(
        {
            "frame": frame,
            "t": (frame - frame[0]) / float(fps),
            "px": px,
            "py": py,
            "p_heading": heading,
            "p_speed": np.asarray(p_speed, dtype=float),
            "tx": tx,
            "ty": ty,
            "tvx": np.asarray(tvx, dtype=float),
            "tvy": np.asarray(tvy, dtype=float),
            "range": np.hypot(px - tx, py - ty),
            "bearing": wrap_angle(los - heading),
        }
    )
    for name, values in optional.items():
        if name not in OPTIONAL_COLUMNS:
            raise ValueError(f"unknown optional column {name!r}")
        df[name] = np.asarray(values, dtype=float)
    return df
