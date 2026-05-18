"""Tests for the chase data contract."""

import numpy as np
import pandas as pd
import pytest

from aor.contracts import (
    Chase,
    ChaseMeta,
    ContractError,
    build_frames,
    validate_chase,
    wrap_angle,
)


def make_chase(n: int = 60, fps: float = 30.0) -> Chase:
    """A pursuer running along +x toward a target that drifts along +y."""
    frame = np.arange(100, 100 + n)
    t = (frame - frame[0]) / fps
    px = 5.0 * t
    py = np.zeros(n)
    tx = np.full(n, 20.0)
    ty = 2.0 * t
    heading = np.zeros(n)
    speed = np.full(n, 5.0)
    df = build_frames(frame, fps, px, py, heading, speed, tx, ty, np.zeros(n), np.full(n, 2.0))
    meta = ChaseMeta(
        dataset="synthetic",
        session_id="s1",
        animal_id="a1",
        chase_id="c1",
        fps=fps,
        start_frame=int(frame[0]),
        end_frame=int(frame[-1]),
        end_reason="clip_end",
        target_type="cricket",
    )
    return Chase(meta=meta, frames=df)


def test_wrap_angle_range():
    a = np.linspace(-10, 10, 1001)
    w = wrap_angle(a)
    assert np.all(w > -np.pi) and np.all(w <= np.pi)
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))
    assert wrap_angle(np.pi) == pytest.approx(np.pi)
    assert wrap_angle(-np.pi) == pytest.approx(np.pi)


def test_valid_chase_passes():
    chase = make_chase()
    validate_chase(chase, min_duration_s=1.0)
    assert chase.duration_s == pytest.approx(2.0)


def test_bearing_geometry():
    chase = make_chase()
    df = chase.frames
    # At t = 0 the target is at (20, 0) and the pursuer at (0, 0) heading +x: bearing 0.
    assert df["bearing"].iloc[0] == pytest.approx(0.0)
    # Later the target has moved to +y, so the bearing is positive (counter-clockwise).
    assert df["bearing"].iloc[-1] > 0


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda df: df.drop(columns=["range"]), "missing required"),
        (lambda df: df.assign(frame=df["frame"].where(df.index != 10, df["frame"] + 1)), "not contiguous"),
        (lambda df: df.assign(px=df["px"].where(df.index != 5, np.nan)), "NaN present"),
        (lambda df: df.assign(range=df["range"] + 1.0), "range column disagrees"),
        (lambda df: df.assign(p_heading=df["p_heading"] + 4.0), "p_heading outside"),
        (lambda df: df.assign(p_speed=df["p_speed"] * 100.0), "pursuer speed exceeds"),
        (lambda df: df.assign(tvx=df["tvx"] + 1000.0), "target speed exceeds"),
        (lambda df: df.assign(t=df["t"] * 2.0), "frame spacing"),
        (lambda df: df.assign(extra=1.0), "unknown columns"),
    ],
)
def test_violations_are_reported(mutate, needle):
    chase = make_chase()
    chase.frames = mutate(chase.frames)
    with pytest.raises(ContractError) as err:
        validate_chase(chase)
    assert needle in str(err.value)


def test_coincident_positions_rejected():
    chase = make_chase()
    df = chase.frames.copy()
    df.loc[3, ["tx", "ty"]] = df.loc[3, ["px", "py"]].to_numpy()
    df["range"] = np.hypot(df["px"] - df["tx"], df["py"] - df["ty"])
    chase.frames = df
    with pytest.raises(ContractError) as err:
        validate_chase(chase)
    assert "coincide" in str(err.value)


def test_meta_checks():
    chase = make_chase()
    chase.meta.end_reason = "whatever"
    chase.meta.target_type = "owl"
    with pytest.raises(ContractError) as err:
        validate_chase(chase)
    msg = str(err.value)
    assert "end_reason" in msg and "target_type" in msg


def test_min_duration():
    chase = make_chase(n=15)
    with pytest.raises(ContractError) as err:
        validate_chase(chase, min_duration_s=1.0)
    assert "below minimum" in str(err.value)


def test_optional_columns_allowed():
    chase = make_chase()
    n = len(chase.frames)
    chase.frames["px_pix"] = chase.frames["px"] * 27.0
    chase.frames["t_speed"] = np.full(n, 2.0)
    validate_chase(chase)
    with pytest.raises(ValueError):
        build_frames(
            np.arange(3),
            30.0,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.ones(3),
            np.ones(3),
            np.zeros(3),
            np.zeros(3),
            nonsense=np.zeros(3),
        )


def test_all_frames_are_floats_after_build():
    chase = make_chase()
    for col in ("t", "px", "py", "p_heading", "p_speed", "tx", "ty", "tvx", "tvy", "range", "bearing"):
        assert pd.api.types.is_float_dtype(chase.frames[col])
    assert pd.api.types.is_integer_dtype(chase.frames["frame"])


def test_native_rate_and_frame_checks():
    chase = make_chase()
    chase.meta.native_fps = 30.0
    validate_chase(chase)
    chase.meta.native_fps = 20.0
    with pytest.raises(ContractError) as err:
        validate_chase(chase)
    assert "exceeds the native rate" in str(err.value)
    chase.meta.native_fps = None
    chase.meta.coordinate_frame = ""
    with pytest.raises(ContractError) as err:
        validate_chase(chase)
    assert "coordinate_frame" in str(err.value)
