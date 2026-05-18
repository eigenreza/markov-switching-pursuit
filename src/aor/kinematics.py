"""Shared kinematic helpers: NaN-aware smoothing, headings from a body axis, velocities.

All functions work on one-dimensional time series or on (T, 2) position arrays sampled at a
fixed frame rate. Missing samples are NaN. Nothing here changes units: callers pass
centimetres and get centimetres per second back.
"""

from __future__ import annotations

import numpy as np

from .contracts import wrap_angle


def moving_average(x: np.ndarray, window: int, min_valid_fraction: float = 0.5) -> np.ndarray:
    """Centred moving average that ignores NaN. Output is NaN where fewer than
    ``min_valid_fraction`` of the window is valid. ``window`` is forced odd."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window)
    valid = np.isfinite(x)
    num = np.convolve(np.where(valid, x, 0.0), kernel, mode="same")
    cnt = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.full_like(x, np.nan)
    ok = cnt >= max(1.0, min_valid_fraction * window)
    out[ok] = num[ok] / cnt[ok]
    return out


def smooth_xy(xy: np.ndarray, window: int) -> np.ndarray:
    """Moving average of each column of a (T, 2) array."""
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([moving_average(xy[:, 0], window), moving_average(xy[:, 1], window)])


def velocity(xy: np.ndarray, fps: float, window: int = 1) -> np.ndarray:
    """Central-difference velocity (T, 2) of a (T, 2) position series, optionally after smoothing.
    Endpoints use one-sided differences. NaN propagates to neighbouring samples."""
    xy = smooth_xy(xy, window) if window > 1 else np.asarray(xy, dtype=float)
    v = np.full_like(xy, np.nan)
    if len(xy) >= 2:
        v[1:-1] = (xy[2:] - xy[:-2]) * (fps / 2.0)
        v[0] = (xy[1] - xy[0]) * fps
        v[-1] = (xy[-1] - xy[-2]) * fps
    return v


def speed(xy: np.ndarray, fps: float, window: int = 1) -> np.ndarray:
    v = velocity(xy, fps, window)
    return np.hypot(v[:, 0], v[:, 1])


def heading_from_axis(rear: np.ndarray, front: np.ndarray, window: int = 1) -> np.ndarray:
    """Heading angle (radians, wrapped) of the body axis pointing from ``rear`` to ``front``,
    for example tail base to neck. Smoothing is applied to the unit axis vector so that the
    average is taken on the circle, not on wrapped angles."""
    rear = np.asarray(rear, dtype=float)
    front = np.asarray(front, dtype=float)
    axis = front - rear
    norm = np.hypot(axis[:, 0], axis[:, 1])
    unit = axis / np.where(norm > 0, norm, np.nan)[:, None]
    if window > 1:
        unit = smooth_xy(unit, window)
    return wrap_angle(np.arctan2(unit[:, 1], unit[:, 0]))


def heading_from_velocity(xy: np.ndarray, fps: float, window: int = 1, min_speed: float = 0.0) -> np.ndarray:
    """Heading from the direction of travel; NaN where the speed is below ``min_speed``."""
    v = velocity(xy, fps, window)
    s = np.hypot(v[:, 0], v[:, 1])
    h = wrap_angle(np.arctan2(v[:, 1], v[:, 0]))
    h[~(s >= min_speed)] = np.nan
    return h


def angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed smallest difference a - b on the circle, in (-pi, pi]."""
    return wrap_angle(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def interpolate_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of length <= ``max_gap``; longer runs stay NaN."""
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    isnan = np.isnan(x)
    if not isnan.any() or isnan.all():
        return x
    idx = np.arange(n)
    d = np.diff(np.concatenate([[0], isnan.astype(int), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    for s, e in zip(starts, ends, strict=True):
        if e - s <= max_gap and s > 0 and e < n:
            x[s:e] = np.interp(idx[s:e], [s - 1, e], [x[s - 1], x[e]])
    return x


def interpolate_gaps_xy(xy: np.ndarray, max_gap: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([interpolate_gaps(xy[:, 0], max_gap), interpolate_gaps(xy[:, 1], max_gap)])


def runs_of_true(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of runs of True."""
    mask = np.asarray(mask, dtype=bool)
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def despike(xy: np.ndarray, fps: float, max_speed: float) -> np.ndarray:
    """Set to NaN any sample whose implied speed from the previous valid sample exceeds
    ``max_speed``. Removes single-frame tracking jumps; longer excursions survive and are
    caught later by the contract."""
    xy = np.asarray(xy, dtype=float).copy()
    n = len(xy)
    last = None
    for i in range(n):
        if not np.all(np.isfinite(xy[i])):
            continue
        if last is not None:
            dt = (i - last) / fps
            if np.hypot(*(xy[i] - xy[last])) / dt > max_speed:
                xy[i] = np.nan
                continue
        last = i
    return xy


def circular_median(h: np.ndarray, window: int = 3) -> np.ndarray:
    """Running circular median of angles: within each odd window the sample with the smallest total
    angular distance to the others. Removes single-frame flips in a tracked yaw without lag."""
    h = np.asarray(h, dtype=float)
    out = h.copy()
    half = window // 2
    n = len(h)
    for i in range(half, n - half):
        w = h[i - half : i + half + 1]
        if not np.all(np.isfinite(w)):
            continue
        d = np.abs(wrap_angle(w[:, None] - w[None, :])).sum(axis=1)
        out[i] = w[int(np.argmin(d))]
    return out


def repair_flips(h: np.ndarray, threshold: float = np.pi / 2) -> np.ndarray:
    """Repair single-frame excursions of a tracked heading: an interior sample that differs from
    both neighbours by more than ``threshold`` while the neighbours agree within ``threshold`` is
    replaced by the circular mean of the neighbours. Every other sample is left as measured, so
    the repair creates no runs of repeated values."""
    h = np.asarray(h, dtype=float)
    out = h.copy()
    if len(h) < 3:
        return out
    prev, cur, nxt = h[:-2], h[1:-1], h[2:]
    ok = np.isfinite(prev) & np.isfinite(cur) & np.isfinite(nxt)
    d_prev = np.abs(wrap_angle(cur - prev))
    d_next = np.abs(wrap_angle(cur - nxt))
    d_nb = np.abs(wrap_angle(nxt - prev))
    flip = ok & (d_prev > threshold) & (d_next > threshold) & (d_nb <= threshold)
    mean = np.arctan2(np.sin(prev) + np.sin(nxt), np.cos(prev) + np.cos(nxt))
    idx = np.flatnonzero(flip) + 1
    out[idx] = mean[flip]
    return out
