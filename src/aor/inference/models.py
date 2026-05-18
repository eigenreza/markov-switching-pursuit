"""Model fits: single laws, the persistence null, and the switching model with plain or
input-dependent transitions; delay profiling; scoring; posteriors; bootstrap.

Emission model for law k at increment s with delay d:
    r_s = dpsi_s - beta_k * x_k(s - d)          (wrapped residual)
    e_s = r_s - rho * r_{s-1}                    (autoregressive residual, shared rho)
    e_s ~ (1 - eps) Normal(0, sigma^2) + eps Normal(0, sigma_out^2)
where x is the bearing error (pure pursuit, beta = gain), the lead-point error (lead pursuit,
beta = gain, lead time tau), or the line-of-sight rotation (proportional navigation, beta = N).
The gain is shared by the two pursuit laws; sigma, rho, eps, sigma_out and the delay are shared
by all laws. The autoregressive term absorbs the smoothness of tracked headings and the
outlier component absorbs tracking glitches; either can be switched off in the settings.
Rows start at s = d + 2 so that every regressor and the lagged residual are defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize_scalar

from ..contracts import Chase, wrap_angle
from ..laws import LAW_INDEX, LAWS
from .features import ChaseArrays, FeatureParams, covariates, design_pair, lead_error, prepare
from .hmm import (
    filtered_batch,
    forward_backward_batch,
    forward_frames,
    mstep_input,
    mstep_plain,
    pad,
    transition_logits,
    viterbi_batch,
)

ManoeuvreParams = FeatureParams
K = len(LAWS)
PURE, LEAD, PN = LAW_INDEX["pure_pursuit"], LAW_INDEX["lead_pursuit"], LAW_INDEX["proportional_navigation"]
P_COV = 4


@dataclass(frozen=True)
class Emissions:
    """Shared emission parameters. ``rho`` is the first-order autoregressive coefficient of the
    residual heading increment, ``eps`` the weight of an outlier component with standard deviation
    ``sigma_out``; both are inactive when the corresponding setting is off."""

    gain: float = 0.5
    tau: float = 0.3
    N: float = 3.0
    sigma: float = 0.1
    rho: float = 0.0
    eps: float = 0.0
    sigma_out: float = 1.0


@dataclass(frozen=True)
class FitSettings:
    delay: int = 2
    start: int | None = None  # first increment index used; None means delay + 2
    tau_bounds: tuple[float, float] = (0.0, 1.5)
    n_iter: int = 60
    tol: float = 1e-4
    n_restarts: int = 2
    seed: int = 0
    l2: float = 1e-2  # ridge on transition weights (input-dependent model)
    manoeuvre: FeatureParams = field(default_factory=FeatureParams)
    ar_residual: bool = True  # autoregressive residual on the heading increment
    outliers: bool = True  # outlier mixture component on the residual
    rho_bounds: tuple[float, float] = (-0.9, 0.95)
    eps_max: float = 0.3
    sigma_out_min: float = 0.3
    persistence: bool = False  # a fourth state with zero predicted increment (post-freeze follow-up)


@dataclass
class Fit:
    model: str  # "null", "single:<law>", "switching:plain", "switching:input"
    emissions: Emissions
    delay: int
    loglik: float
    n_frames: int
    pi: np.ndarray | None = None
    A: np.ndarray | None = None  # plain transitions (K, K)
    W: np.ndarray | None = None  # input-dependent weights (K, K, P)
    law: str | None = None
    iterations: int = 0
    converged: bool = True
    persistence: bool = False

    @property
    def n_params(self) -> int:
        n = 4  # sigma, rho, eps, sigma_out
        if self.model.startswith("single"):
            n += {"pure_pursuit": 1, "lead_pursuit": 2, "proportional_navigation": 1}[self.law]
        elif self.model.startswith("switching"):
            Kf = K + int(self.persistence)
            n += 3 + (Kf - 1)
            n += Kf * (Kf - 1) if self.A is not None else Kf * (Kf - 1) * self.W.shape[2]
        return n

    def summary(self) -> dict:
        d = {
            "model": self.model,
            "delay": self.delay,
            "loglik": self.loglik,
            "n_frames": self.n_frames,
            "iterations": self.iterations,
            "converged": self.converged,
            "n_params": self.n_params,
            "persistence": self.persistence,
            **{k: float(v) for k, v in self.emissions.__dict__.items()},
        }
        if self.A is not None:
            d["A"] = self.A.tolist()
        if self.W is not None:
            d["W"] = self.W.tolist()
        if self.pi is not None:
            d["pi"] = self.pi.tolist()
        if self.model.startswith("switching"):
            d["state_roles"] = state_roles(self)
        return d


# ----------------------------------------------------------------------------- data access


def arrays(chases: list, mp: FeatureParams = FeatureParams()) -> list[ChaseArrays]:
    return [c if isinstance(c, ChaseArrays) else prepare(c, mp) for c in chases]


def _designs(arrs: list[ChaseArrays], delay: int, tau: float, start: int | None):
    """Per chase: (arrays, y, X, s, y_prev, X_prev). Rows start at max(start, delay + 2) so that
    every model is scored on the same frames."""
    out = []
    for a in arrs:
        y, X, s, yp, Xp = design_pair(a, delay, tau, start)
        if len(y) >= 1:
            out.append((a, y, X, s, yp, Xp))
    return out


# ----------------------------------------------------------------------------- emissions


def predicted_means(X: np.ndarray, em: Emissions, persistence: bool = False) -> np.ndarray:
    """(n, K) predicted heading increments of each law, plus a zero column for the persistence
    state when it is part of the library."""
    m = np.column_stack([em.gain * X[:, PURE], em.gain * X[:, LEAD], em.N * X[:, PN]])
    if persistence:
        m = np.column_stack([m, np.zeros(len(X))])
    return m


def innovations(y, X, yp, Xp, em: Emissions, persistence: bool = False) -> np.ndarray:
    """(n, K) residual innovations e = r_s - rho * r_{s-1}, with r the wrapped residual of each state."""
    r = wrap_angle(y[:, None] - predicted_means(X, em, persistence))
    rp = wrap_angle(yp[:, None] - predicted_means(Xp, em, persistence))
    return r - em.rho * rp


def _log_mixture(e: np.ndarray, em: Emissions) -> np.ndarray:
    core = -0.5 * np.log(2 * np.pi * em.sigma**2) - e**2 / (2 * em.sigma**2)
    if em.eps <= 0:
        return core
    out = -0.5 * np.log(2 * np.pi * em.sigma_out**2) - e**2 / (2 * em.sigma_out**2)
    return np.logaddexp(np.log1p(-em.eps) + core, np.log(em.eps) + out)


def outlier_weight(e: np.ndarray, em: Emissions) -> np.ndarray:
    """Posterior weight of the core (non-outlier) component for each innovation."""
    if em.eps <= 0:
        return np.ones_like(e)
    core = np.log1p(-em.eps) - 0.5 * np.log(2 * np.pi * em.sigma**2) - e**2 / (2 * em.sigma**2)
    out = np.log(em.eps) - 0.5 * np.log(2 * np.pi * em.sigma_out**2) - e**2 / (2 * em.sigma_out**2)
    return np.exp(core - np.logaddexp(core, out))


def log_emissions(y, X, yp, Xp, em: Emissions, persistence: bool = False) -> np.ndarray:
    """(n, K) log densities of the residual innovation under each state."""
    return _log_mixture(innovations(y, X, yp, Xp, em, persistence), em)


def null_log_emissions(y: np.ndarray, yp: np.ndarray, em: Emissions) -> np.ndarray:
    e = wrap_angle(y) - em.rho * wrap_angle(yp)
    return _log_mixture(e, em)


def _weighted_coef(xs, ys, ws) -> float:
    num = sum(float(np.sum(w * x * y)) for x, y, w in zip(xs, ys, ws, strict=True))
    den = sum(float(np.sum(w * x * x)) for x, w in zip(xs, ws, strict=True))
    return num / den if den > 0 else 0.0


def _core_weights(data, gammas, em: Emissions, persistence: bool = False) -> list[np.ndarray]:
    """Per chase (n, K) weights: state responsibility times the core-component weight."""
    out = []
    for (_, y, X, _, yp, Xp), g in zip(data, gammas, strict=True):
        e = innovations(y, X, yp, Xp, em, persistence)
        out.append(g * outlier_weight(e, em))
    return out


def mstep_emissions(
    arrs: list[ChaseArrays],
    delay: int,
    start: int | None,
    gammas: list[np.ndarray],
    em: Emissions,
    settings: FitSettings,
    passes: int = 2,
) -> Emissions:
    """Coordinate updates: gain (pooled over the pursuit laws) and N by weighted least squares on
    the autoregressively transformed regressors, tau by a bounded search, rho by weighted
    regression of the residual on its lag, then sigma, the outlier weight and its scale."""
    tau_bounds = settings.tau_bounds
    for _ in range(passes):
        data = _designs(arrs, delay, em.tau, start)
        W = _core_weights(data, gammas, em, settings.persistence)
        rho = em.rho
        ys = [y - rho * yp for _, y, _, _, yp, _ in data]
        xs_g = [
            np.concatenate([X[:, PURE] - rho * Xp[:, PURE], X[:, LEAD] - rho * Xp[:, LEAD]])
            for _, _, X, _, _, Xp in data
        ]
        ys_g = [np.concatenate([yt, yt]) for yt in ys]
        ws_g = [np.concatenate([w[:, PURE], w[:, LEAD]]) for w in W]
        gain = _weighted_coef(xs_g, ys_g, ws_g)
        N = _weighted_coef(
            [X[:, PN] - rho * Xp[:, PN] for _, _, X, _, _, Xp in data], ys, [w[:, PN] for w in W]
        )
        em = replace(em, gain=gain, N=N)

        def sse(tau, em=em, data=data, W=W):
            tot = 0.0
            for (a, y, _, s, yp, _), w in zip(data, W, strict=True):
                le = lead_error(a, tau)
                xl, xlp = le[s - delay], le[s - delay - 1]
                e = wrap_angle(y - em.gain * xl) - em.rho * wrap_angle(yp - em.gain * xlp)
                tot += float(np.sum(w[:, LEAD] * e**2))
            return tot

        if sum(float(w[:, LEAD].sum()) for w in W) > 1e-6:
            res = minimize_scalar(sse, bounds=tau_bounds, method="bounded", options={"xatol": 1e-3})
            em = replace(em, tau=float(res.x))
        data = _designs(arrs, delay, em.tau, start)
        if settings.ar_residual:
            num = den = 0.0
            for (_, y, X, _, yp, Xp), w in zip(data, W, strict=True):
                r = wrap_angle(y[:, None] - predicted_means(X, em, settings.persistence))
                rp = wrap_angle(yp[:, None] - predicted_means(Xp, em, settings.persistence))
                num += float(np.sum(w * r * rp))
                den += float(np.sum(w * rp * rp))
            rho = float(np.clip(num / den if den > 0 else 0.0, *settings.rho_bounds))
            em = replace(em, rho=rho)
        num = den = num_o = den_o = tot = 0.0
        for (_, y, X, _, yp, Xp), g in zip(data, gammas, strict=True):
            e = innovations(y, X, yp, Xp, em, settings.persistence)
            wc = outlier_weight(e, em) if settings.outliers else np.ones_like(e)
            num += float(np.sum(g * wc * e**2))
            den += float(np.sum(g * wc))
            num_o += float(np.sum(g * (1 - wc) * e**2))
            den_o += float(np.sum(g * (1 - wc)))
            tot += float(np.sum(g))
        sigma = float(np.sqrt(max(num / max(den, 1e-12), 1e-6)))
        if settings.outliers:
            eps = float(np.clip(den_o / max(tot, 1e-12), 1e-4, settings.eps_max))
            sigma_out = float(np.sqrt(max(num_o / max(den_o, 1e-12), settings.sigma_out_min**2)))
            em = replace(em, sigma=sigma, eps=eps, sigma_out=sigma_out)
        else:
            em = replace(em, sigma=sigma, eps=0.0)
    return em


# ----------------------------------------------------------------------------- single-law and null fits


def fit_null(chases: list, settings: FitSettings = FitSettings()) -> Fit:
    """Persistence null: increments are noise around zero, with the same residual structure
    (autoregressive residual and outlier mixture) as the laws."""
    arrs = arrays(chases, settings.manoeuvre)
    data = _designs(arrs, settings.delay, 0.0, settings.start)
    y = np.concatenate([d[1] for d in data])
    yp = np.concatenate([d[4] for d in data])
    em = Emissions(
        gain=0.0,
        tau=0.0,
        N=0.0,
        sigma=float(np.sqrt(np.mean(wrap_angle(y) ** 2))),
        rho=0.0,
        eps=0.0,
        sigma_out=1.0,
    )
    prev = -np.inf
    for _ in range(settings.n_iter):
        e = wrap_angle(y) - em.rho * wrap_angle(yp)
        wc = outlier_weight(e, em) if settings.outliers else np.ones_like(e)
        if settings.ar_residual:
            num = float(np.sum(wc * wrap_angle(y) * wrap_angle(yp)))
            den = float(np.sum(wc * wrap_angle(yp) ** 2))
            em = replace(em, rho=float(np.clip(num / den if den > 0 else 0.0, *settings.rho_bounds)))
            e = wrap_angle(y) - em.rho * wrap_angle(yp)
        sigma = float(np.sqrt(max(np.sum(wc * e**2) / max(wc.sum(), 1e-12), 1e-6)))
        if settings.outliers:
            eps = float(np.clip((1 - wc).sum() / len(wc), 1e-4, settings.eps_max))
            sigma_out = float(
                np.sqrt(max(np.sum((1 - wc) * e**2) / max((1 - wc).sum(), 1e-12), settings.sigma_out_min**2))
            )
            em = replace(em, sigma=sigma, eps=eps, sigma_out=sigma_out)
        else:
            em = replace(em, sigma=sigma)
        ll = float(np.sum(null_log_emissions(y, yp, em)))
        if abs(ll - prev) < settings.tol * max(1.0, abs(ll)):
            prev = ll
            break
        prev = ll
    return Fit("null", em, settings.delay, prev, len(y))


def fit_single(
    chases: list, law: str, settings: FitSettings = FitSettings(), init: Emissions = Emissions()
) -> Fit:
    """One law for every frame: the switching model with the posterior pinned to that law."""
    settings = replace(settings, persistence=False)  # single-law fits never carry the persistence state
    arrs = arrays(chases, settings.manoeuvre)
    k = LAW_INDEX[law]
    em = init
    data = _designs(arrs, settings.delay, em.tau, settings.start)
    gammas = [np.eye(K)[k][None, :].repeat(len(y), axis=0) for _, y, _, _, _, _ in data]
    prev = -np.inf
    it = 0
    for it in range(1, settings.n_iter + 1):  # noqa: B007
        em = mstep_emissions(arrs, settings.delay, settings.start, gammas, em, settings, passes=1)
        data = _designs(arrs, settings.delay, em.tau, settings.start)
        ll = sum(float(np.sum(log_emissions(y, X, yp, Xp, em)[:, k])) for _, y, X, _, yp, Xp in data)
        if abs(ll - prev) < settings.tol * max(1.0, abs(ll)):
            prev = ll
            break
        prev = ll
    n = sum(len(y) for _, y, _, _, _, _ in data)
    return Fit(f"single:{law}", em, settings.delay, prev, n, law=law, iterations=it)


# ----------------------------------------------------------------------------- switching fits


def _batch_inputs(data, em: Emissions, pi, A, W, persistence: bool = False):
    """Padded emissions, lengths, and log transitions for a batch of chases."""
    lbs = [log_emissions(y, X, yp, Xp, em, persistence) for _, y, X, _, yp, Xp in data]
    log_b, lengths = pad(lbs, fill=0.0)
    if W is not None:
        Zs = [covariates(a, s)[1:] for a, _, _, s, _, _ in data]  # covariates of the frame transitioned into
        Z, _ = pad([z if len(z) else np.zeros((0, P_COV)) for z in Zs], fill=0.0)
        if Z.shape[1] == 0:
            Z = np.zeros((len(data), 0, P_COV))
        log_A = transition_logits(W, Z)  # (n, T-1, K, K)
        return log_b, lengths, np.log(pi), log_A, Z
    return log_b, lengths, np.log(pi), np.log(A), None


def _estep(data, em: Emissions, pi, A, W, persistence: bool = False):
    log_b, lengths, log_pi, log_A, Z = _batch_inputs(data, em, pi, A, W, persistence)
    gamma, xi, ll = forward_backward_batch(log_b, lengths, log_pi, log_A)
    gammas = [gamma[i, : lengths[i]] for i in range(len(data))]
    return gammas, gamma, xi, Z, lengths, float(ll.sum())


def fit_switching(
    chases: list,
    settings: FitSettings = FitSettings(),
    transitions: str = "plain",
    init: Emissions | None = None,
    init_A: np.ndarray | None = None,
) -> Fit:
    """Expectation-maximisation for the switching model. ``transitions`` is "plain" (a fixed
    matrix) or "input" (logistic in range, closing speed and recent manoeuvre). Restarts from
    perturbed starting points guard against poor local optima."""
    arrs = arrays(chases, settings.manoeuvre)
    rng = np.random.default_rng(settings.seed)
    pers = settings.persistence
    Kf = K + int(pers)
    if init is None:
        pure = fit_single(arrs, "pure_pursuit", settings)
        pn = fit_single(arrs, "proportional_navigation", settings)
        lead = fit_single(arrs, "lead_pursuit", settings, init=replace(pure.emissions, tau=0.3))
        init = Emissions(
            gain=pure.emissions.gain,
            tau=lead.emissions.tau,
            N=pn.emissions.N,
            sigma=min(pure.emissions.sigma, lead.emissions.sigma, pn.emissions.sigma),
            rho=pure.emissions.rho,
            eps=pure.emissions.eps,
            sigma_out=pure.emissions.sigma_out,
        )
    best: Fit | None = None
    for r in range(max(1, settings.n_restarts)):
        em = (
            init
            if r == 0
            else replace(
                init,
                gain=init.gain * rng.uniform(0.7, 1.3),
                tau=float(np.clip(init.tau * rng.uniform(0.5, 1.5), *settings.tau_bounds)),
                N=init.N * rng.uniform(0.7, 1.3),
                sigma=init.sigma * rng.uniform(0.8, 1.2),
            )
        )
        stay = 0.9 if r == 0 else rng.uniform(0.8, 0.98)
        A = (
            init_A
            if (init_A is not None and r == 0 and init_A.shape == (Kf, Kf))
            else (np.full((Kf, Kf), (1 - stay) / (Kf - 1)) + (stay - (1 - stay) / (Kf - 1)) * np.eye(Kf))
        )
        W = None
        A_use = A
        if transitions == "input":
            W = np.zeros((Kf, Kf, P_COV))
            off = ~np.eye(Kf, dtype=bool)
            W[off, 0] = np.log(A[off] / np.diag(A).repeat(Kf - 1))
            A_use = None
        pi = np.full(Kf, 1.0 / Kf)
        prev = -np.inf
        it = 0
        converged = False
        data = _designs(arrs, settings.delay, em.tau, settings.start)
        for it in range(1, settings.n_iter + 1):
            data = _designs(arrs, settings.delay, em.tau, settings.start)
            gammas, gamma, xi, Z, lengths, ll = _estep(data, em, pi, A_use, W, pers)
            if ll - prev < settings.tol * max(1.0, abs(ll)) and it > 2:
                converged = True
                prev = ll
                break
            prev = ll
            pi, A_new = mstep_plain(xi, gamma, lengths)
            if transitions == "input":
                W = mstep_input(xi, Z, W, settings.l2)
            else:
                A_use = A_new
            em = mstep_emissions(arrs, settings.delay, settings.start, gammas, em, settings, passes=1)
        n = sum(len(y) for _, y, _, _, _, _ in data)
        fit = Fit(
            f"switching:{transitions}",
            em,
            settings.delay,
            prev,
            n,
            pi=pi,
            A=A_use,
            W=W,
            iterations=it,
            converged=converged,
            persistence=pers,
        )
        if best is None or fit.loglik > best.loglik:
            best = fit
    return best


# ----------------------------------------------------------------------------- scoring and posteriors


def score(chases: list, fit: Fit, start: int | None = None, mp: FeatureParams = FeatureParams()) -> dict:
    """Log-likelihood of chases under a fitted model: total, per chase, and the frame count.
    ``start`` fixes the first increment used so that models with different delays are scored
    on the same frames."""
    arrs = arrays(chases, mp)
    data = _designs(arrs, fit.delay, fit.emissions.tau, start if start is not None else fit.delay + 2)
    per = {}
    if fit.model.startswith("switching"):
        log_b, lengths, log_pi, log_A, _ = _batch_inputs(
            data, fit.emissions, fit.pi, fit.A, fit.W, fit.persistence
        )
        _, _, ll = forward_backward_batch(log_b, lengths, log_pi, log_A)
        for (a, y, _, _, _, _), lc in zip(data, ll, strict=True):
            per[a.chase_id] = (float(lc), len(y))
    else:
        for a, y, X, _, yp, Xp in data:
            if fit.model == "null":
                lc = float(np.sum(null_log_emissions(y, yp, fit.emissions)))
            else:
                lc = float(np.sum(log_emissions(y, X, yp, Xp, fit.emissions)[:, LAW_INDEX[fit.law]]))
            per[a.chase_id] = (lc, len(y))
    tot = sum(v[0] for v in per.values())
    n = sum(v[1] for v in per.values())
    return {"loglik": tot, "n_frames": n, "per_frame": tot / max(n, 1), "per_chase": per}


def score_frames(
    chases: list, fit: Fit, start: int | None = None, mp: FeatureParams = FeatureParams()
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-frame log-likelihood contributions of each chase under a fitted model, as
    {chase_id: (increment indices s, log-likelihood per increment)}. For the switching models the
    contributions are the forward predictive terms, which sum to the marginal log-likelihood."""
    arrs = arrays(chases, mp)
    data = _designs(arrs, fit.delay, fit.emissions.tau, start if start is not None else fit.delay + 2)
    out = {}
    if fit.model.startswith("switching"):
        log_b, lengths, log_pi, log_A, _ = _batch_inputs(
            data, fit.emissions, fit.pi, fit.A, fit.W, fit.persistence
        )
        ff = forward_frames(log_b, lengths, log_pi, log_A)
        for i, (a, _y, _, s, _, _) in enumerate(data):
            out[a.chase_id] = (s, ff[i, : lengths[i]])
    else:
        for a, y, X, s, yp, Xp in data:
            if fit.model == "null":
                out[a.chase_id] = (s, null_log_emissions(y, yp, fit.emissions))
            else:
                out[a.chase_id] = (s, log_emissions(y, X, yp, Xp, fit.emissions)[:, LAW_INDEX[fit.law]])
    return out


def posteriors(chases: list, fit: Fit, mp: FeatureParams = FeatureParams()) -> list[dict]:
    """Posterior law probabilities (n, K) and the Viterbi path for every chase, with the
    increment indices they refer to and the true law where known."""
    arrs = arrays(chases, mp)
    data = _designs(arrs, fit.delay, fit.emissions.tau, fit.delay + 2)
    out = []
    if fit.model.startswith("switching"):
        log_b, lengths, log_pi, log_A, _ = _batch_inputs(
            data, fit.emissions, fit.pi, fit.A, fit.W, fit.persistence
        )
        gamma, _, _ = forward_backward_batch(log_b, lengths, log_pi, log_A)
        filt = filtered_batch(log_b, lengths, log_pi, log_A)
        paths = viterbi_batch(log_b, lengths, log_pi, log_A)
        for i, (a, _y, _, s, _, _) in enumerate(data):
            L = lengths[i]
            out.append(
                {
                    "chase_id": a.chase_id,
                    "s": s,
                    "gamma": gamma[i, :L],
                    "filtered": filt[i, :L],
                    "path": paths[i, :L],
                    "law_true": None if a.law_true is None else a.law_true[s],
                }
            )
    else:
        for a, y, _, s, _, _ in data:
            g = np.zeros((len(y), K))
            k = LAW_INDEX[fit.law] if fit.law else PURE
            g[:, k] = 1.0
            out.append(
                {
                    "chase_id": a.chase_id,
                    "s": s,
                    "gamma": g,
                    "filtered": g,
                    "path": np.full(len(y), k),
                    "law_true": None if a.law_true is None else a.law_true[s],
                }
            )
    return out


ROLE_THRESHOLD = 0.05  # per-frame response below which a fitted state is called non-steering


def state_roles(fit: Fit) -> list[str]:
    """Role of each state of a switching fit: "steering" when the state's fitted response to the
    target is at least ROLE_THRESHOLD per frame (the gain for pure and lead pursuit, the navigation
    constant for proportional navigation), "non-steering" otherwise and for the persistence state."""
    em = fit.emissions
    roles = ["steering" if abs(em.gain) >= ROLE_THRESHOLD else "non-steering"] * 2
    roles.append("steering" if abs(em.N) >= ROLE_THRESHOLD else "non-steering")
    if fit.persistence:
        roles.append("non-steering")
    return roles


def steering_probability(gamma: np.ndarray, fit: Fit) -> np.ndarray:
    """Posterior probability that the pursuer is in a steering state."""
    cols = [k for k, r in enumerate(state_roles(fit)) if r == "steering"]
    return gamma[:, cols].sum(axis=1) if cols else np.zeros(len(gamma))


# ----------------------------------------------------------------------------- delay profile and bootstrap


def delay_profile(
    chases: list,
    delays: list[int],
    settings: FitSettings = FitSettings(),
    model: str = "switching:plain",
    law: str | None = None,
) -> list[dict]:
    """Fit the model at each delay, scoring every fit on the common support that starts after
    the largest delay. Returns a list of {delay, loglik, n_frames, fit}."""
    start = max(delays) + 2
    out = []
    for d in delays:
        st = replace(settings, delay=d, start=start)
        if model == "null":
            f = fit_null(chases, st)
        elif model.startswith("single"):
            f = fit_single(chases, law or model.split(":")[1], st)
        else:
            f = fit_switching(chases, st, transitions=model.split(":")[1])
        out.append({"delay": d, "loglik": f.loglik, "n_frames": f.n_frames, "fit": f})
    return out


def best_delay(profile: list[dict]) -> int:
    return int(max(profile, key=lambda r: r["loglik"])["delay"])


def bootstrap(
    chases: list, fit_fn, B: int = 100, seed: int = 0, strata: list[str] | None = None
) -> list[Fit]:
    """Chase-level bootstrap, resampling whole chases within strata (for example animals).
    ``fit_fn`` maps a list of chases to a Fit."""
    rng = np.random.default_rng(seed)
    chases = list(chases)
    if strata is None:
        strata = [c.meta.animal_id if isinstance(c, Chase) else c.animal_id for c in chases]
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    fits = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups.values()])
        fits.append(fit_fn([chases[i] for i in idx]))
    return fits


def interval(values, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    lo = (1 - level) / 2
    return float(np.quantile(values, lo)), float(np.quantile(values, 1 - lo))
