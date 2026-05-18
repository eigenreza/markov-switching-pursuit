"""Inference tests: hand-computed two-frame likelihoods, forward-backward against brute force,
and recovery of laws and parameters on short simulations."""

import itertools

import numpy as np
import pytest

from aor.contracts import Chase, ChaseMeta, build_frames
from aor.inference.features import ManoeuvreParams, manoeuvre_indicator, prepare
from aor.inference.hmm import forward_backward, mstep_plain, transition_logits, viterbi
from aor.inference.models import (
    Emissions,
    FitSettings,
    best_delay,
    bootstrap,
    delay_profile,
    fit_null,
    fit_single,
    fit_switching,
    interval,
    log_emissions,
    posteriors,
    score,
)
from aor.laws import LAW_INDEX, LawParams
from aor.simulator import PursuerParams, SwitchParams, TargetParams, WorldParams, simulate, to_chase

FPS = 30.0


def two_increment_chase() -> Chase:
    """Four frames: a pursuer at rest at the origin, heading 0, 0, 0, 0.1; a target fixed at (10, 10)."""
    n = 4
    frame = np.arange(n)
    px = np.zeros(n)
    py = np.zeros(n)
    heading = np.array([0.0, 0.0, 0.0, 0.1])
    tx = np.full(n, 10.0)
    ty = np.full(n, 10.0)
    df = build_frames(frame, FPS, px, py, heading, np.zeros(n), tx, ty, np.zeros(n), np.zeros(n))
    meta = ChaseMeta("synthetic", "s", "a", "fixture", FPS, 0, n - 1, "clip_end", "simulated", native_fps=FPS)
    return Chase(meta=meta, frames=df)


def test_two_frame_likelihood_by_hand():
    chase = two_increment_chase()
    arr = prepare(chase)
    # with delay 0 the support is the single increment s = 2: dpsi = 0.1, bearing = pi/4, dlam = 0
    from aor.inference.features import design_pair

    y, X, s, yp, Xp = design_pair(arr, delay=0, tau=0.5)
    assert s.tolist() == [2]
    assert y[0] == pytest.approx(0.1) and yp[0] == pytest.approx(0.0)
    assert X[0, LAW_INDEX["pure_pursuit"]] == pytest.approx(np.pi / 4)
    assert X[0, LAW_INDEX["lead_pursuit"]] == pytest.approx(
        np.pi / 4
    )  # target at rest: lead point is the target
    assert X[0, LAW_INDEX["proportional_navigation"]] == pytest.approx(0.0)
    em = Emissions(gain=0.5, tau=0.5, N=3.0, sigma=0.2, rho=0.0, eps=0.0)
    ll = log_emissions(y, X, yp, Xp, em)[0]
    const = -0.5 * np.log(2 * np.pi * 0.04)
    # pure pursuit: mean 0.5 * pi/4 = 0.392699, residual -0.292699, squared 0.085673, over 0.08
    assert ll[LAW_INDEX["pure_pursuit"]] == pytest.approx(const - 0.085673 / 0.08, abs=1e-4)
    assert ll[LAW_INDEX["lead_pursuit"]] == pytest.approx(ll[LAW_INDEX["pure_pursuit"]])
    # proportional navigation: mean 0, residual 0.1
    assert ll[LAW_INDEX["proportional_navigation"]] == pytest.approx(const - 0.01 / 0.08, abs=1e-9)
    assert const == pytest.approx(0.690499, abs=1e-5)


def brute_force(log_b, log_pi, log_A):
    T, K = log_b.shape
    total = {}
    for path in itertools.product(range(K), repeat=T):
        lp = log_pi[path[0]] + log_b[0, path[0]]
        for t in range(1, T):
            lp += log_A[path[t - 1], path[t]] + log_b[t, path[t]]
        total[path] = lp
    logz = np.log(sum(np.exp(v) for v in total.values()))
    gamma = np.zeros((T, K))
    for path, lp in total.items():
        for t, k in enumerate(path):
            gamma[t, k] += np.exp(lp - logz)
    best = max(total, key=total.get)
    return gamma, logz, np.array(best)


def test_forward_backward_and_viterbi_match_brute_force():
    rng = np.random.default_rng(0)
    T, K = 5, 3
    log_b = np.log(rng.uniform(0.1, 1.0, (T, K)))
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K) * 2, size=K)
    g, xi, ll = forward_backward(log_b, np.log(pi), np.log(A))
    g_ref, ll_ref, path_ref = brute_force(log_b, np.log(pi), np.log(A))
    assert ll == pytest.approx(ll_ref, abs=1e-9)
    assert np.allclose(g, g_ref, atol=1e-9)
    assert np.allclose(xi.sum(axis=2), g[:-1], atol=1e-9)
    assert np.allclose(xi.sum(axis=1), g[1:], atol=1e-9)
    assert np.array_equal(viterbi(log_b, np.log(pi), np.log(A)), path_ref)
    # time-varying transitions give the same answers when every matrix is the same
    g2, _, ll2 = forward_backward(log_b, np.log(pi), np.broadcast_to(np.log(A), (T - 1, K, K)).copy())
    assert ll2 == pytest.approx(ll_ref, abs=1e-9) and np.allclose(g2, g_ref, atol=1e-9)


def test_mstep_plain_counts():
    K = 2
    gamma = np.zeros((1, 3, K))
    gamma[0] = [[1, 0], [0, 1], [0, 1]]
    xi = np.zeros((1, 2, K, K))
    xi[0, 0] = [[0, 1], [0, 0]]
    xi[0, 1] = [[0, 0], [0, 1]]
    pi, A = mstep_plain(xi, gamma, np.array([3]), prior_stay=0.0)
    assert pi[0] > 0.99
    assert A[0, 1] > 0.99 and A[1, 1] > 0.99


def test_transition_logits_reference_class():
    K, P = 3, 2
    W = np.zeros((K, K, P))
    Z = np.array([[1.0, 0.0]])
    logA = transition_logits(W, Z)
    assert np.allclose(np.exp(logA[0]), np.full((K, K), 1 / 3))
    W[0, 1, 0] = 2.0  # raise 0 -> 1
    logA = transition_logits(W, Z)
    assert np.exp(logA[0, 0, 1]) > np.exp(logA[0, 0, 0])
    assert np.allclose(np.exp(logA[0]).sum(axis=1), 1.0)


def test_manoeuvre_indicator_detects_turn_and_jump():
    n = 40
    tvx = np.full(n, 10.0)
    tvy = np.zeros(n)
    tvy[20:] = 10.0  # 45 degree turn at frame 20
    tvx[20:] = 0.0  # ... actually a 90 degree turn
    man = manoeuvre_indicator(tvx, tvy, FPS, ManoeuvreParams(lag_frames=3, heading_change_deg=60.0))
    hits = np.flatnonzero(man)
    assert len(hits) == 1 and 20 <= hits[0] <= 23
    tvx2 = np.full(n, 1.0)
    tvx2[25:28] = 60.0  # a jump: speed spike
    man2 = manoeuvre_indicator(tvx2, np.zeros(n), FPS, ManoeuvreParams())
    assert man2.any() and 25 <= np.flatnonzero(man2)[0] <= 28


def sim_set(law, params, n=12, target=None, switching=SwitchParams(), seed0=100, max_frames=150, speed=18.0):
    tgt = target or TargetParams(kind="constant_velocity", speed=12.0, turn_rate_hz=0.6, turn_sd=1.0)
    out = []
    for i in range(n):
        res = simulate(
            tgt,
            PursuerParams(law=law, params=params, speed_mean=speed),
            switching,
            WorldParams(max_frames=max_frames, arena=(80.0, 80.0)),
            seed=seed0 + i,
        )
        c = to_chase(res, f"{law}_{i}", animal_id=f"a{i % 3}")
        if len(c.frames) >= 20:
            out.append(c)
    return out


def test_single_law_fit_recovers_parameters():
    chases = sim_set("lead_pursuit", LawParams(gain=0.5, delay=2, noise=0.05, tau=0.5), n=16)
    fit = fit_single(chases, "lead_pursuit", FitSettings(delay=2))
    assert fit.emissions.gain == pytest.approx(0.5, abs=0.07)
    assert fit.emissions.tau == pytest.approx(0.5, abs=0.15)
    assert fit.emissions.sigma == pytest.approx(0.05, abs=0.02)
    pn = sim_set("proportional_navigation", LawParams(N=3.0, delay=1, noise=0.05), n=16, seed0=300)
    fit_pn = fit_single(pn, "proportional_navigation", FitSettings(delay=1))
    assert pytest.approx(3.0, abs=0.4) == fit_pn.emissions.N


def test_single_law_comparison_identifies_generating_law():
    for law, params in [
        ("pure_pursuit", LawParams(gain=0.5, delay=2, noise=0.05)),
        ("lead_pursuit", LawParams(gain=0.5, delay=2, noise=0.05, tau=0.6)),
        ("proportional_navigation", LawParams(N=3.0, delay=2, noise=0.05)),
    ]:
        chases = sim_set(law, params, n=10, seed0=500)
        lls = {k: fit_single(chases, k, FitSettings(delay=2)).loglik for k in LAW_INDEX}
        assert max(lls, key=lls.get) == law, lls
    null = fit_null(chases, FitSettings(delay=2))
    assert null.loglik < max(lls.values())


def test_delay_profile_peaks_at_true_delay():
    chases = sim_set("pure_pursuit", LawParams(gain=0.5, delay=3, noise=0.05), n=12, seed0=700)
    prof = delay_profile(chases, [0, 1, 2, 3, 4, 5], FitSettings(), model="single:pure_pursuit")
    assert best_delay(prof) == 3
    assert all(np.isfinite(r["loglik"]) for r in prof)


def test_switching_fit_recovers_fixed_switch():
    sw = SwitchParams(kind="fixed", fixed=((0, "lead_pursuit"), (50, "pure_pursuit")))
    chases = sim_set(
        "lead_pursuit",
        LawParams(gain=0.5, delay=2, noise=0.05, tau=0.6),
        n=14,
        switching=sw,
        seed0=900,
        max_frames=120,
    )
    fit = fit_switching(chases, FitSettings(delay=2, n_iter=40, n_restarts=1), transitions="plain")
    assert np.isfinite(fit.loglik)
    post = posteriors(chases, fit)
    correct = np.concatenate([(p["path"] == p["law_true"]).astype(float) for p in post])
    assert correct.mean() > 0.75
    # the switching model explains the data better than any single law on these frames
    best_single = max(fit_single(chases, k, FitSettings(delay=2)).loglik for k in LAW_INDEX)
    assert fit.loglik > best_single
    sc = score(chases, fit)
    assert sc["loglik"] == pytest.approx(fit.loglik, rel=1e-6)


def test_input_dependent_transitions_run_and_fit():
    tgt = TargetParams(kind="jumper", jump_rate_hz=1.0, jump_length=6.0, crawl_speed=1.0)
    sw = SwitchParams(
        kind="manoeuvre_triggered", base_law="lead_pursuit", reaction_law="pure_pursuit", reaction_frames=10
    )
    chases = sim_set(
        "lead_pursuit",
        LawParams(gain=0.5, delay=2, noise=0.05, tau=0.5),
        n=12,
        target=tgt,
        switching=sw,
        seed0=1100,
        max_frames=200,
        speed=10.0,
    )
    plain = fit_switching(chases, FitSettings(delay=2, n_iter=30, n_restarts=1), transitions="plain")
    inp = fit_switching(
        chases,
        FitSettings(delay=2, n_iter=30, n_restarts=1),
        transitions="input",
        init=plain.emissions,
        init_A=plain.A,
    )
    assert np.isfinite(inp.loglik)
    assert inp.W is not None and inp.W.shape == (3, 3, 4)
    assert inp.loglik >= plain.loglik - 1e-3 * abs(plain.loglik)
    # the recent-manoeuvre covariate raises the lead -> pure transition
    w = inp.W[LAW_INDEX["lead_pursuit"], LAW_INDEX["pure_pursuit"], 3]
    assert w > 0


def test_bootstrap_and_interval():
    chases = sim_set("pure_pursuit", LawParams(gain=0.5, delay=2, noise=0.05), n=9, seed0=1300)
    fits = bootstrap(chases, lambda cs: fit_single(cs, "pure_pursuit", FitSettings(delay=2)), B=8, seed=1)
    gains = [f.emissions.gain for f in fits]
    lo, hi = interval(gains)
    assert lo <= np.median(gains) <= hi
    assert len(fits) == 8


def test_causal_velocity_matches_backward_difference():
    from aor.inference.features import causal_velocity

    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    vx, _vy = causal_velocity(x, np.zeros(5), 30.0, 2)
    assert vx[2] == pytest.approx((3.0 - 0.0) * 30.0 / 2)
    assert vx[4] == pytest.approx((10.0 - 3.0) * 30.0 / 2)
    assert vx[0] == vx[2] and vx[1] == vx[2]
