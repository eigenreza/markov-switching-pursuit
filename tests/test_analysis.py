"""Tests for the real-data analysis pieces on synthetic chases: decimation, the generic switching
comparator, the evader library arrays, the optimal-play benchmark, and the two-player test."""

import numpy as np
import pandas as pd

from aor.analysis import optimal, twoplayer
from aor.analysis.datasets import decimate, informative_mask
from aor.contracts import validate_chase
from aor.inference.comparator import fit_switching_ar, score_frames_switching_ar, score_switching_ar
from aor.inference.features import FeatureParams, prepare_evader
from aor.inference.models import (
    FitSettings,
    fit_single,
    fit_switching,
    posteriors,
    score,
    score_frames,
    state_roles,
    steering_probability,
)
from aor.laws import LawParams
from aor.simulator import (
    PursuerParams,
    SwitchParams,
    TargetParams,
    WorldParams,
    simulate,
    simulate_along_target,
    to_chase,
)


def chases(
    n=10, seed0=50, law="pure_pursuit", params=LawParams(gain=0.3, delay=1, noise=0.05), max_frames=120
):
    out = []
    for i in range(n):
        res = simulate(
            TargetParams(kind="constant_velocity", speed=10.0, turn_rate_hz=0.6),
            PursuerParams(law=law, params=params, speed_mean=16.0),
            SwitchParams(),
            WorldParams(arena=(80.0, 80.0), max_frames=max_frames),
            seed=seed0 + i,
        )
        c = to_chase(res, f"c{i}", animal_id=f"a{i % 2}")
        if len(c.frames) >= 20:
            out.append(c)
    return out


def test_decimate_keeps_contract_and_rate():
    c = chases(1)[0]
    d = decimate(c, 3)
    validate_chase(d)
    assert d.meta.fps == c.meta.fps / 3
    assert len(d.frames) == len(c.frames.iloc[::3])
    assert np.allclose(d.frames["px"].to_numpy(), c.frames["px"].to_numpy()[::3])


def test_informative_mask_uses_target_speed():
    c = chases(1)[0]
    m = informative_mask(c, v_min=3.0)
    assert m.dtype == bool and len(m) == len(c.frames)
    assert m.mean() > 0.5  # the target moves at 10 cm/s


def test_switching_ar_fits_and_scores_consistently():
    cs = chases(8)
    fit = fit_switching_ar(cs, delay=1, n_iter=20)
    assert np.isfinite(fit.loglik) and fit.a.shape == (3,) and fit.emissions.sigma > 0
    sc = score_switching_ar(cs, fit)
    frames = score_frames_switching_ar(cs, fit)
    total = sum(v[1].sum() for v in frames.values())
    assert abs(total - sc["loglik"]) < 1e-6 * max(1.0, abs(sc["loglik"]))


def test_score_frames_sum_to_score():
    cs = chases(6)
    st = FitSettings(delay=1, n_iter=20, n_restarts=1)
    for fit in (fit_single(cs, "pure_pursuit", st), fit_switching(cs, st)):
        sc = score(cs, fit)
        fr = score_frames(cs, fit)
        total = sum(v[1].sum() for v in fr.values())
        assert abs(total - sc["loglik"]) < 1e-6 * max(1.0, abs(sc["loglik"]))


def test_evader_arrays_point_away():
    c = chases(1)[0]
    # give the target a heading column equal to its direction of travel
    f = c.frames
    f["t_heading"] = np.arctan2(f["tvy"], f["tvx"])
    arr = prepare_evader(c, FeatureParams())
    assert arr.away
    # the pure-evasion error is the angle from the evader's heading to the direction away from the pursuer
    away = np.arctan2(f["ty"] - f["py"], f["tx"] - f["px"])
    assert np.allclose(
        np.angle(np.exp(1j * (arr.bearing - (away - f["t_heading"].to_numpy())))), 0.0, atol=1e-9
    )
    assert np.allclose(arr.dlam, 0.0)


def test_simulate_along_target_tracks_and_captures():
    c = chases(1, params=LawParams(gain=0.5, delay=0, noise=0.0))[0]
    f = c.frames
    res = simulate_along_target(
        f["tx"].to_numpy(),
        f["ty"].to_numpy(),
        c.meta.fps,
        float(f["px"].iloc[0]),
        float(f["py"].iloc[0]),
        float(f["p_heading"].iloc[0]),
        f["p_speed"].to_numpy(),
        "pure_pursuit",
        LawParams(gain=0.5, delay=0, noise=0.0),
        capture_radius=3.0,
        collision_course=True,
    )
    assert len(res["px"]) == len(f)
    assert res["capture_frame"] is None or res["capture_frame"] <= len(f) - 1


def test_optimal_benchmark_outputs():
    cs = chases(5)
    dist, cap, prof = optimal.run(cs, delay=1)
    assert len(dist) == 5 and (dist["distance_rad"] >= 0).all() and (dist["distance_rad"] <= np.pi).all()
    assert (dist["distance_lo"] <= dist["distance_hi"]).all()
    assert set(cap.columns) >= {"optimal_capture_s", "actual_capture_s", "end_reason"}
    assert set(prof["variable"]) == {"phase", "range", "t_speed"}


def _frames_from_posteriors(post, fit):
    roles = np.array([r == "steering" for r in state_roles(fit)])
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "chase_id": p["chase_id"],
                    "frame": p["s"],
                    "p_pure": p["gamma"][:, 0],
                    "p_lead": p["gamma"][:, 1],
                    "p_pn": p["gamma"][:, 2],
                    "p_steering": steering_probability(p["gamma"], fit),
                    "p_steering_filtered": steering_probability(p["filtered"], fit),
                    "path": p["path"],
                    "steering_path": roles[p["path"]].astype(int),
                    "steering_filtered": roles[np.argmax(p["filtered"], axis=1)].astype(int),
                }
            )
            for p in post
        ]
    )


def test_manoeuvre_test_runs_on_synthetic_posteriors():
    cs = chases(8, params=LawParams(gain=0.3, delay=1, noise=0.05))
    fit = fit_switching(cs, FitSettings(delay=1, n_iter=15, n_restarts=1))
    post = posteriors(cs, fit)
    frames = _frames_from_posteriors(post, fit)
    rows = twoplayer.manoeuvre_rows(cs, frames, FeatureParams())
    assert {"manoeuvre", "range", "closing", "p_steering_prev", "steering_prev"} <= set(rows.columns)
    rep = twoplayer.manoeuvre_test(rows, B=20, n_perm=20)
    assert "p_steering_prev" in rep["coefficients"] and 0 < rep["permutation_p_steering"] <= 1
    est = rep["coefficients"]["p_steering_prev"]["estimate"]
    assert abs(rep["within_chase_effect"] - (est - rep["permutation_null_mean"])) < 1e-12


def test_persistence_state_and_roles():
    cs = chases(8, params=LawParams(gain=0.3, delay=1, noise=0.05))
    st = FitSettings(delay=1, n_iter=15, n_restarts=1, persistence=True)
    fit = fit_switching(cs, st)
    assert fit.persistence and fit.A.shape == (4, 4) and len(fit.pi) == 4
    assert fit.n_params == 4 + 3 + 3 + 12
    roles = state_roles(fit)
    assert len(roles) == 4 and roles[3] == "non-steering"
    post = posteriors(cs, fit)
    g = post[0]["gamma"]
    assert g.shape[1] == 4 and np.allclose(g.sum(axis=1), 1.0)
    ps = steering_probability(g, fit)
    assert np.all(ps >= 0) and np.all(ps <= 1)
    sc = score(cs, fit)
    fr = score_frames(cs, fit)
    total = sum(v[1].sum() for v in fr.values())
    assert abs(total - sc["loglik"]) < 1e-6 * max(1.0, abs(sc["loglik"]))
    # a three-state fit from the same chases carries three roles and ignores a four-state init matrix
    fit3 = fit_switching(cs, FitSettings(delay=1, n_iter=5, n_restarts=1), init=fit.emissions, init_A=fit.A)
    assert fit3.A.shape == (3, 3) and len(state_roles(fit3)) == 3


def test_comparator_state_count():
    cs = chases(6)
    fit4 = fit_switching_ar(cs, delay=1, n_iter=10, n_restarts=1, n_states=4)
    assert fit4.n_states == 4 and fit4.A.shape == (4, 4) and fit4.n_params == 8 + 4 + 3 + 12
    sc = score_switching_ar(cs, fit4)
    assert np.isfinite(sc["loglik"])


def test_logistic_fit_recovers_sign():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(4000), rng.normal(size=4000)])
    p = 1 / (1 + np.exp(-(-1.0 + 1.5 * X[:, 1])))
    y = (rng.random(4000) < p).astype(float)
    w = twoplayer.logistic_fit(X, y)
    assert abs(w[0] + 1.0) < 0.2 and abs(w[1] - 1.5) < 0.2


def test_cross_correlation_and_occupancy_shapes():
    rng = np.random.default_rng(1)
    rows_p, rows_e = [], []
    for cid in range(6):
        n = 60
        x = rng.random(n)
        rows_p.append(
            pd.DataFrame(
                {
                    "chase_id": f"c{cid}",
                    "frame": np.arange(n),
                    "p_pure": 1 - x,
                    "p_lead": x * 0.7,
                    "p_pn": x * 0.3,
                    "p_steering": x,
                    "path": (x > 0.5).astype(int),
                    "steering_path": (x > 0.5).astype(int),
                }
            )
        )
        rows_e.append(
            pd.DataFrame(
                {
                    "chase_id": f"c{cid}",
                    "frame": np.arange(n),
                    "e_pure": 1 - x,
                    "e_lead": x,
                    "e_straight": 0.0,
                    "e_path": (x > 0.5).astype(int),
                }
            )
        )
    P, E = pd.concat(rows_p), pd.concat(rows_e)
    xc = twoplayer.cross_correlation(P, E, max_lag=5, n_shuffle=20)
    assert len(xc) == 11 and xc.loc[xc["lag_frames"] == 0, "correlation"].iloc[0] > 0.9
    jo = twoplayer.joint_occupancy(P, E)
    assert abs(jo[jo["phase_bin"] == -1]["fraction"].sum() - 1.0) < 1e-9


def test_arena_fit_scale_matches_extent():
    from aor.data.michaiel import ARENA_CM, PX_PER_CM, Trial, arena_fit_scale

    rng = np.random.default_rng(0)
    xy = np.column_stack([rng.uniform(100, 100 + 1440, 20000), rng.uniform(50, 50 + 805, 20000)]) / PX_PER_CM
    tr = Trial(0, "a", "1", "0", xy, xy.copy(), np.zeros(len(xy)), np.zeros(len(xy)))
    sx, sy = arena_fit_scale([tr])
    assert abs(sx - 1440 / ARENA_CM[0]) < 0.5 and abs(sy - 805 / ARENA_CM[1]) < 0.5


def test_filtered_posteriors_are_causal_and_normalised():
    from aor.inference.hmm import filtered_batch, forward_backward_batch, pad

    rng = np.random.default_rng(0)
    log_b = [np.log(rng.random((n, 3))) for n in (12, 7)]
    lb, lengths = pad(log_b)
    log_pi = np.log(np.full(3, 1 / 3))
    A = np.full((3, 3), 0.1) + 0.7 * np.eye(3)
    f = filtered_batch(lb, lengths, log_pi, np.log(A))
    g, _, _ = forward_backward_batch(lb, lengths, log_pi, np.log(A))
    assert np.allclose(f[0].sum(axis=1), 1.0) and np.allclose(f[1, :7].sum(axis=1), 1.0)
    assert np.allclose(f[0, -1], g[0, -1])  # at the last frame filtering and smoothing agree
    # changing a later observation changes the smoothed but not the filtered probability at frame 3
    lb2 = lb.copy()
    lb2[0, 8] = np.log(np.array([0.9, 0.05, 0.05]))
    f2 = filtered_batch(lb2, lengths, log_pi, np.log(A))
    g2, _, _ = forward_backward_batch(lb2, lengths, log_pi, np.log(A))
    assert np.allclose(f2[0, 3], f[0, 3]) and not np.allclose(g2[0, 3], g[0, 3])


def test_state_regression_separates_a_steering_state():
    from aor.analysis.fit_real import state_regression

    cs = chases(8, params=LawParams(gain=0.3, delay=1, noise=0.05))
    fit = fit_switching(cs, FitSettings(delay=1, n_iter=15, n_restarts=1))
    sr = state_regression(cs, fit, FeatureParams())
    assert list(sr["state"]) == ["pure", "lead", "pn"]
    assert abs(sr["share"].sum() - 1.0) < 1e-9
    steering = sr[sr["n_frames"] >= 3]
    assert (steering["slope"].abs() < 2).all()


def test_result_csv_keeps_the_null_model_name(tmp_path):
    from aor import paper, tables
    from aor.figures import real

    path = tmp_path / "c.csv"
    pd.DataFrame({"model": ["null", "switching:plain"], "x": [1.0, float("nan")]}).to_csv(path, index=False)
    for read in (tables._csv, paper._csv, real._read):
        df = read(path)
        assert df["model"].tolist() == ["null", "switching:plain"] and df["x"].isna().tolist() == [
            False,
            True,
        ]
