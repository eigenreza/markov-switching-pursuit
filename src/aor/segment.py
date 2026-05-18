"""Chase segmentation from pursuer and target kinematics.

The rule follows the preprocessing contract: a chase starts when the pursuer begins
approaching (range decreasing, pursuer speed above a threshold, target within a bearing
window) from at least a minimum range, and ends at contact (range below a contact radius)
or when the range has stopped decreasing for a stall window, or when the track is lost.

Thresholds live in ``SegmentationParams``. The values used in the audit are provisional;
the frozen values are given in docs/data.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import angular_difference, moving_average, runs_of_true


@dataclass(frozen=True)
class SegmentationParams:
    min_pursuer_speed: float = 10.0  # cm/s
    min_closing_speed: float = 5.0  # cm/s, rate of range decrease that counts as approaching
    max_bearing_deg: float = 60.0  # target must lie within this angle of the pursuer heading at start
    min_start_range: float = 8.0  # cm
    contact_range: float = 3.0  # cm, ends the chase as contact
    stall_window_s: float = 0.5  # chase ends when range has not decreased over this window
    min_duration_s: float = 0.5
    smooth_window_s: float = 0.1  # smoothing applied to range and speed before thresholds
    max_gap_frames: int = 5  # approach runs separated by at most this many frames are joined


@dataclass
class Segment:
    start: int  # inclusive frame index into the input arrays
    end: int  # exclusive
    end_reason: str  # contact, stall, track_loss, clip_end
    start_range: float
    end_range: float

    @property
    def n_frames(self) -> int:
        return self.end - self.start


def segment_chases(
    px: np.ndarray,
    py: np.ndarray,
    heading: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    fps: float,
    params: SegmentationParams = SegmentationParams(),
) -> list[Segment]:
    """Return chase segments for one continuous recording (NaN allowed in the inputs)."""
    px, py, tx, ty, heading = (np.asarray(a, dtype=float) for a in (px, py, tx, ty, heading))
    n = len(px)
    w = max(1, round(params.smooth_window_s * fps))
    rng = np.hypot(tx - px, ty - py)
    rng_s = moving_average(rng, w)
    closing = -np.gradient(rng_s) * fps
    pxs, pys = moving_average(px, w), moving_average(py, w)
    spd = np.hypot(np.gradient(pxs), np.gradient(pys)) * fps
    los = np.arctan2(ty - py, tx - px)
    bearing = np.abs(angular_difference(los, heading))
    valid = np.isfinite(rng) & np.isfinite(spd) & np.isfinite(closing)

    approaching = valid & (closing > params.min_closing_speed) & (spd > params.min_pursuer_speed)
    approaching &= np.isfinite(bearing) & (bearing < np.deg2rad(params.max_bearing_deg))
    starts, ends = runs_of_true(approaching)
    for i in range(len(starts) - 1):
        if starts[i + 1] - ends[i] <= params.max_gap_frames:
            approaching[ends[i] : starts[i + 1]] = True
    starts, ends = runs_of_true(approaching)

    stall = max(1, round(params.stall_window_s * fps))
    segments: list[Segment] = []
    cursor = 0
    for s, e in zip(starts, ends, strict=True):
        if s < cursor:
            continue
        if not (np.isfinite(rng_s[s]) and rng_s[s] >= params.min_start_range):
            continue
        j = e
        reason = "clip_end"
        while True:
            if j >= n:
                j = n
                reason = "clip_end"
                break
            if not valid[j]:
                reason = "track_loss"
                break
            if rng[j] <= params.contact_range:
                j += 1
                reason = "contact"
                break
            lo = max(s, j - stall)
            if j - lo >= stall and np.nanmin(rng_s[lo : j + 1]) >= rng_s[lo] - 1e-9 and not approaching[j]:
                reason = "stall"
                break
            j += 1
        if (j - s) / fps >= params.min_duration_s:
            segments.append(Segment(int(s), int(j), reason, float(rng[s]), float(rng[j - 1])))
        cursor = j
    return segments
