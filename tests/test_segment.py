import numpy as np

from aor.segment import SegmentationParams, segment_chases


def pursuit_scene(fps=30.0, n=120, pursuer_speed=20.0, start_range=30.0, target_speed=0.0):
    """Pursuer runs along +x toward a target ahead that drifts along +x at ``target_speed``."""
    t = np.arange(n) / fps
    px = pursuer_speed * t
    py = np.zeros(n)
    tx = start_range + target_speed * t
    ty = np.zeros(n)
    heading = np.zeros(n)
    return px, py, heading, tx, ty


def test_detects_single_chase_ending_in_contact():
    fps = 30.0
    px, py, h, tx, ty = pursuit_scene(fps=fps)
    segs = segment_chases(px, py, h, tx, ty, fps, SegmentationParams())
    assert len(segs) == 1
    s = segs[0]
    assert s.end_reason == "contact"
    assert s.start_range >= 8.0
    assert s.end_range <= 3.0
    assert s.n_frames / fps >= 0.5


def test_no_chase_when_pursuer_is_slow():
    fps = 30.0
    px, py, h, tx, ty = pursuit_scene(fps=fps, pursuer_speed=3.0)
    assert segment_chases(px, py, h, tx, ty, fps) == []


def test_no_chase_when_target_is_behind():
    fps = 30.0
    px, py, h, tx, ty = pursuit_scene(fps=fps)
    h = np.full_like(h, np.pi)  # pursuer faces -x while moving +x: target is behind its heading
    assert segment_chases(px, py, h, tx, ty, fps) == []


def test_stall_ends_chase():
    fps = 30.0
    n = 150
    t = np.arange(n) / fps
    # approach for 1.5 s, then stop 10 cm short and stay for 3.5 s
    px = np.where(t < 1.5, 20.0 * t, 30.0)
    py = np.zeros(n)
    tx = np.full(n, 40.0)
    ty = np.zeros(n)
    h = np.zeros(n)
    segs = segment_chases(px, py, h, tx, ty, fps)
    assert len(segs) == 1
    assert segs[0].end_reason == "stall"
    assert segs[0].end < n - 15  # ended well before the clip end


def test_track_loss_ends_chase():
    fps = 30.0
    px, py, h, tx, ty = pursuit_scene(fps=fps, n=120)
    tx = tx.copy()
    tx[40:] = np.nan
    segs = segment_chases(px, py, h, tx, ty, fps)
    assert len(segs) == 1
    assert segs[0].end_reason == "track_loss"
    assert segs[0].end <= 41


def test_min_duration_filter():
    fps = 30.0
    px, py, h, tx, ty = pursuit_scene(fps=fps, n=10, pursuer_speed=60.0, start_range=12.0)
    assert segment_chases(px, py, h, tx, ty, fps, SegmentationParams(min_duration_s=1.0)) == []
