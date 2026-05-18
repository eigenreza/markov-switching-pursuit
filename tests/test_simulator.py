import numpy as np
import pytest

from aor.contracts import validate_chase
from aor.laws import LAW_INDEX, LawParams
from aor.simulator import PursuerParams, SwitchParams, TargetParams, WorldParams, simulate, to_chase

WORLD = WorldParams(fps=30.0, arena=(45.0, 38.0), capture_radius=2.0, max_frames=400)


def still_target():
    return TargetParams(kind="constant_velocity", speed=0.0, turn_rate_hz=0.0, heading_noise=0.0)


def test_pure_pursuit_captures_still_target_in_expected_time():
    p = PursuerParams(
        law="pure_pursuit", params=LawParams(gain=0.5, delay=0, noise=0.0), speed_mean=20.0, speed_sd=0.0
    )
    res = simulate(still_target(), p, SwitchParams(), WORLD, seed=1, start=(5.0, 19.0, 0.0, 35.0, 19.0, 0.0))
    assert res.end_reason == "capture"
    # 30 cm to cover minus the capture radius at 20 cm/s: 1.4 s = 42 frames
    assert abs(res.n_frames - 43) <= 2


def test_proportional_navigation_nulls_los_rate_against_constant_velocity_target():
    tgt = TargetParams(kind="constant_velocity", speed=8.0, turn_rate_hz=0.0, heading_noise=0.0)
    p = PursuerParams(
        law="proportional_navigation",
        params=LawParams(N=4.0, delay=0, noise=0.0),
        speed_mean=20.0,
        speed_sd=0.0,
    )
    res = simulate(tgt, p, SwitchParams(), WORLD, seed=2, start=(5.0, 5.0, 0.3, 30.0, 10.0, np.pi / 2))
    assert res.end_reason == "capture"
    los = np.arctan2(res.ty - res.py, res.tx - res.px)
    dlos = np.abs(np.diff(los[10:-5]))
    assert np.median(dlos) < 0.01  # line of sight nearly fixed in direction mid-chase


def test_lead_pursuit_beats_pure_pursuit_against_crossing_target():
    tgt = TargetParams(kind="constant_velocity", speed=10.0, turn_rate_hz=0.0, heading_noise=0.0)
    start = (5.0, 5.0, 0.3, 30.0, 8.0, np.pi / 2)
    pure = simulate(
        tgt,
        PursuerParams(
            law="pure_pursuit", params=LawParams(gain=0.6, noise=0.0), speed_mean=16.0, speed_sd=0.0
        ),
        SwitchParams(),
        WORLD,
        seed=3,
        start=start,
    )
    lead = simulate(
        tgt,
        PursuerParams(
            law="lead_pursuit", params=LawParams(gain=0.6, tau=0.8, noise=0.0), speed_mean=16.0, speed_sd=0.0
        ),
        SwitchParams(),
        WORLD,
        seed=3,
        start=start,
    )
    assert pure.end_reason == "capture" and lead.end_reason == "capture"
    assert lead.n_frames < pure.n_frames


def test_delay_applies_command_from_earlier_frame():
    p = PursuerParams(
        law="pure_pursuit", params=LawParams(gain=0.5, delay=3, noise=0.0), speed_mean=10.0, speed_sd=0.0
    )
    res = simulate(still_target(), p, SwitchParams(), WORLD, seed=4, start=(5.0, 5.0, 1.0, 30.0, 30.0, 0.0))
    # for the first three frames nothing is applied; afterwards the heading increment equals the delayed command
    assert np.allclose(res.command[:3], 0.0)
    dh = np.diff(res.heading)
    assert np.allclose(np.angle(np.exp(1j * (dh - res.command[:-1]))), 0.0, atol=1e-9)


def test_jumper_makes_manoeuvres_and_displacements():
    tgt = TargetParams(kind="jumper", jump_rate_hz=2.0, jump_length=6.0, jump_frames=3, crawl_speed=0.0)
    p = PursuerParams(law="pure_pursuit", params=LawParams(gain=0.3, noise=0.0), speed_mean=5.0, speed_sd=0.0)
    world = WorldParams(fps=30.0, arena=(45.0, 38.0), capture_radius=1.0, max_frames=300)
    res = simulate(tgt, p, SwitchParams(), world, seed=5, start=(5.0, 19.0, 0.0, 40.0, 19.0, 0.0))
    assert res.manoeuvre.sum() >= 5
    # a jump spreads 6 cm over 3 frames: per-frame displacement of 2 cm on jump frames
    step = np.hypot(np.diff(res.tx), np.diff(res.ty))
    assert np.isclose(np.max(step), 2.0, atol=0.3)


def test_evader_keeps_range_longer_than_still_target():
    p = PursuerParams(
        law="pure_pursuit", params=LawParams(gain=0.5, noise=0.0), speed_mean=14.0, speed_sd=0.0
    )
    start = (5.0, 19.0, 0.0, 25.0, 19.0, 0.0)
    still = simulate(still_target(), p, SwitchParams(), WORLD, seed=6, start=start)
    ev = simulate(
        TargetParams(kind="evader", speed=8.0, evade_range=15.0, turn_rate_hz=0.0, heading_noise=0.0),
        p,
        SwitchParams(),
        WORLD,
        seed=6,
        start=start,
    )
    assert ev.n_frames > still.n_frames


def test_fixed_switching_records_true_law():
    sw = SwitchParams(kind="fixed", fixed=((0, "lead_pursuit"), (20, "pure_pursuit")))
    p = PursuerParams(law="lead_pursuit", params=LawParams(gain=0.4, tau=0.5, noise=0.02), speed_mean=12.0)
    res = simulate(still_target(), p, sw, WORLD, seed=7, start=(5.0, 5.0, 0.0, 40.0, 30.0, 0.0))
    assert set(res.law_index[:20]) == {LAW_INDEX["lead_pursuit"]}
    assert set(res.law_index[20:]) == {LAW_INDEX["pure_pursuit"]}


def test_manoeuvre_triggered_switching_reacts_after_jumps():
    tgt = TargetParams(kind="jumper", jump_rate_hz=1.0, jump_length=5.0, crawl_speed=0.0)
    sw = SwitchParams(
        kind="manoeuvre_triggered", base_law="lead_pursuit", reaction_law="pure_pursuit", reaction_frames=8
    )
    p = PursuerParams(
        law="lead_pursuit", params=LawParams(gain=0.3, tau=0.3, noise=0.0), speed_mean=6.0, speed_sd=0.0
    )
    res = simulate(
        tgt,
        p,
        sw,
        WorldParams(max_frames=300, capture_radius=1.0),
        seed=8,
        start=(5.0, 19.0, 0.0, 40.0, 19.0, 0.0),
    )
    jumps = np.flatnonzero(res.manoeuvre)
    assert len(jumps) > 0
    j = jumps[0]
    assert res.law_index[j + 1] == LAW_INDEX["pure_pursuit"]
    assert res.law_index[j + 9] in (LAW_INDEX["lead_pursuit"], LAW_INDEX["pure_pursuit"])
    assert res.law_index[0] == LAW_INDEX["lead_pursuit"]


def test_markov_switching_uses_matrix():
    sw = SwitchParams(kind="markov", transition=((0.9, 0.1, 0.0), (0.1, 0.9, 0.0), (0.0, 0.0, 1.0)))
    p = PursuerParams(law="pure_pursuit", params=LawParams(gain=0.3, noise=0.0), speed_mean=5.0, speed_sd=0.0)
    res = simulate(
        still_target(),
        p,
        sw,
        WorldParams(max_frames=400, capture_radius=0.5),
        seed=9,
        start=(1.0, 1.0, 0.0, 44.0, 37.0, 0.0),
    )
    assert set(res.law_index) <= {0, 1}
    assert (np.diff(res.law_index) != 0).sum() > 3


def test_export_passes_contract_and_is_reproducible():
    tgt = TargetParams(kind="constant_velocity", speed=6.0, turn_rate_hz=1.0)
    p = PursuerParams(
        law="lead_pursuit", params=LawParams(gain=0.4, tau=0.4, delay=2, noise=0.05), speed_mean=15.0
    )
    a = to_chase(simulate(tgt, p, SwitchParams(), WORLD, seed=11), "c1")
    b = to_chase(simulate(tgt, p, SwitchParams(), WORLD, seed=11), "c1")
    validate_chase(a, min_duration_s=0.5)
    assert "law_true" in a.frames.columns
    assert a.meta.target_type == "simulated" and a.meta.native_fps == 30.0
    assert a.frames.equals(b.frames)
    c = to_chase(simulate(tgt, p, SwitchParams(), WORLD, seed=12), "c2")
    assert not a.frames.equals(c.frames)


def test_measurement_noise_option():
    tgt = TargetParams(kind="constant_velocity", speed=6.0, turn_rate_hz=0.5)
    p = PursuerParams(law="pure_pursuit", params=LawParams(gain=0.4, noise=0.05), speed_mean=15.0)
    noisy = WorldParams(obs_pos_sd=0.3, obs_heading_sd=0.05)
    res = simulate(tgt, p, SwitchParams(), noisy, seed=13)
    chase = to_chase(res, "noisy")
    validate_chase(chase)
    assert not np.allclose(chase.frames["px"].to_numpy(), res.px)


def test_unknown_target_kind_rejected():
    with pytest.raises(ValueError):
        simulate(TargetParams(kind="teleporter"), PursuerParams(), SwitchParams(), WORLD, seed=0)
