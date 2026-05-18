"""A generic switching comparator with no law library: a switching autoregressive mean on the
heading increment with the same number of discrete states as the law library and the same noise
model as the law models (one shared residual scale, one shared autoregressive residual
coefficient and one shared outlier component).

State k: dpsi_s = a_k + b_k * dpsi_{s-1} + r_s, where the residual r_s carries the shared
first-order autoregressive structure and outlier mixture of the law models. Fitted by
expectation-maximisation with the same forward-backward machinery as the law model and scored on
the same frames, it tests whether the law library adds anything beyond generic segmentation of
the steering signal. Sharing the noise model keeps the comparison to the mean structure: a
comparator with a free residual scale per state is also rewarded for isolating frames whose
tracked heading repeats exactly, which the law models cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..contracts import wrap_angle
from .features import FeatureParams, support
from .hmm import forward_backward_batch, forward_frames, mstep_plain, pad
from .models import Emissions, FitSettings, _log_mixture, arrays, outlier_weight

K = 3  # default number of states, the size of the law library
B_INIT = np.array([0.0, 0.5, -0.5, 0.25, -0.25, 0.75])  # starting autoregressive coefficients per state


@dataclass
class ARFit:
    model: str
    delay: int
    a: np.ndarray  # (K,) intercepts
    b: np.ndarray  # (K,) autoregressive coefficients of the mean
    emissions: Emissions  # sigma, rho, eps and sigma_out are used; the law parameters are zero
    pi: np.ndarray
    A: np.ndarray
    loglik: float
    n_frames: int
    iterations: int = 0

    @property
    def n_states(self) -> int:
        return len(self.a)

    @property
    def n_params(self) -> int:
        k = self.n_states
        return 2 * k + 4 + (k - 1) + k * (k - 1)

    def summary(self) -> dict:
        em = self.emissions
        return {
            "model": self.model,
            "delay": self.delay,
            "n_states": self.n_states,
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "sigma": em.sigma,
            "rho": em.rho,
            "eps": em.eps,
            "sigma_out": em.sigma_out,
            "pi": self.pi.tolist(),
            "A": self.A.tolist(),
            "loglik": self.loglik,
            "n_frames": self.n_frames,
            "n_params": self.n_params,
            "iterations": self.iterations,
        }


def _rows(arrs, delay: int, start: int | None):
    """Per chase: (arrays, s, y, x, y_prev, x_prev) with y = dpsi[s], x = dpsi[s - 1] and the
    pair of the previous row, on the same support as the law models (s from max(start, delay + 2))."""
    out = []
    for a in arrs:
        s = support(a, delay, max(start if start is not None else delay + 2, 2))
        if len(s) >= 1:
            d = wrap_angle(a.dpsi)
            out.append((a, s, d[s], d[s - 1], d[s - 1], d[s - 2]))
    return out


def _innovations(y, x, yp, xp, a: np.ndarray, b: np.ndarray, em: Emissions) -> np.ndarray:
    """(n, K) innovations e = r_s - rho * r_{s-1}, with r the residual from the mean of each state."""
    r = wrap_angle(y[:, None] - a[None, :] - b[None, :] * x[:, None])
    rp = wrap_angle(yp[:, None] - a[None, :] - b[None, :] * xp[:, None])
    return r - em.rho * rp


def _log_b(y, x, yp, xp, a, b, em: Emissions) -> np.ndarray:
    return _log_mixture(_innovations(y, x, yp, xp, a, b, em), em)


def _fit_once(data, delay: int, n_iter: int, tol: float, rng, settings: FitSettings, k: int) -> ARFit:
    ys = np.concatenate([d[2] for d in data])
    s0 = float(np.sqrt(np.mean(ys**2)))
    a = rng.normal(0.0, 0.5 * s0, k)
    b = B_INIT[:k] + rng.normal(0, 0.1, k)
    em = Emissions(
        gain=0.0,
        tau=0.0,
        N=0.0,
        sigma=s0,
        rho=0.0,
        eps=0.05 if settings.outliers else 0.0,
        sigma_out=max(3 * s0, settings.sigma_out_min),
    )
    stay = 0.85
    A = np.full((k, k), (1 - stay) / (k - 1)) + (stay - (1 - stay) / (k - 1)) * np.eye(k)
    pi = np.full(k, 1.0 / k)
    prev = -np.inf
    it = 0
    for it in range(1, n_iter + 1):
        log_b, lengths = pad([_log_b(y, x, yp, xp, a, b, em) for _, _, y, x, yp, xp in data])
        gamma, xi, ll = forward_backward_batch(log_b, lengths, np.log(pi), np.log(A))
        ll = float(ll.sum())
        if ll - prev < tol * max(1.0, abs(ll)) and it > 2:
            prev = ll
            break
        prev = ll
        pi, A = mstep_plain(xi, gamma, lengths)
        # responsibilities times the core-component weight of every innovation
        weights = []
        for i, (_, _, y, x, yp, xp) in enumerate(data):
            e = _innovations(y, x, yp, xp, a, b, em)
            weights.append(gamma[i, : len(y)] * outlier_weight(e, em))
        # per-state intercept and slope by weighted least squares on the quasi-differenced rows
        rho = em.rho
        for j in range(k):
            S = np.zeros((2, 2))
            t = np.zeros(2)
            for (_, _, y, x, yp, xp), w in zip(data, weights, strict=True):
                z = np.column_stack([np.full(len(y), 1.0 - rho), x - rho * xp])
                v = y - rho * yp
                wk = w[:, j]
                S += z.T @ (z * wk[:, None])
                t += z.T @ (wk * v)
            if S[0, 0] > 1e-9 and np.linalg.det(S) > 1e-12:
                a[j], b[j] = np.linalg.solve(S, t)
        # shared autoregressive coefficient of the residual
        if settings.ar_residual:
            num = den = 0.0
            for (_, _, y, x, yp, xp), w in zip(data, weights, strict=True):
                r = wrap_angle(y[:, None] - a[None, :] - b[None, :] * x[:, None])
                rp = wrap_angle(yp[:, None] - a[None, :] - b[None, :] * xp[:, None])
                num += float(np.sum(w * r * rp))
                den += float(np.sum(w * rp**2))
            em = replace(em, rho=float(np.clip(num / den if den > 0 else 0.0, *settings.rho_bounds)))
        # shared scale and outlier component
        core_sse = core_w = out_sse = out_w = tot_w = 0.0
        for i, (_, _, y, x, yp, xp) in enumerate(data):
            e = _innovations(y, x, yp, xp, a, b, em)
            g = gamma[i, : len(y)]
            wc = outlier_weight(e, em) if settings.outliers else np.ones_like(e)
            core_sse += float(np.sum(g * wc * e**2))
            core_w += float(np.sum(g * wc))
            out_sse += float(np.sum(g * (1 - wc) * e**2))
            out_w += float(np.sum(g * (1 - wc)))
            tot_w += float(g.sum())
        sigma = float(np.sqrt(max(core_sse / max(core_w, 1e-12), 1e-6)))
        if settings.outliers:
            eps = float(np.clip(out_w / max(tot_w, 1e-12), 1e-4, settings.eps_max))
            sigma_out = float(np.sqrt(max(out_sse / max(out_w, 1e-12), settings.sigma_out_min**2)))
            em = replace(em, sigma=sigma, eps=eps, sigma_out=sigma_out)
        else:
            em = replace(em, sigma=sigma)
    n = sum(len(d[2]) for d in data)
    return ARFit("switching_ar", delay, a, b, em, pi, A, prev, n, it)


def fit_switching_ar(
    chases: list,
    delay: int,
    start: int | None = None,
    n_iter: int = 60,
    tol: float = 1e-4,
    seed: int = 0,
    mp: FeatureParams = FeatureParams(),
    settings: FitSettings | None = None,
    n_restarts: int = 2,
    n_states: int = K,
) -> ARFit:
    """Fit the comparator on the same support as the law models; ``settings`` supplies the noise
    model flags and bounds (autoregressive residual, outlier mixture) and defaults to the defaults
    of the law models. ``n_states`` matches the size of the library it is compared with. The best
    of ``n_restarts`` random starts is returned."""
    st = settings if settings is not None else FitSettings(delay=delay)
    arrs = arrays(chases, mp)
    data = _rows(arrs, delay, start)
    best = None
    for r in range(max(1, n_restarts)):
        rng = np.random.default_rng(seed + r)
        fit = _fit_once(data, delay, n_iter, tol, rng, st, int(n_states))
        if best is None or fit.loglik > best.loglik:
            best = fit
    return best


def _scored(chases: list, fit: ARFit, start: int | None, mp: FeatureParams):
    arrs = arrays(chases, mp)
    data = _rows(arrs, fit.delay, start if start is not None else fit.delay + 2)
    log_b, lengths = pad([_log_b(y, x, yp, xp, fit.a, fit.b, fit.emissions) for _, _, y, x, yp, xp in data])
    return data, log_b, lengths


def score_frames_switching_ar(
    chases: list, fit: ARFit, start: int | None = None, mp: FeatureParams = FeatureParams()
) -> dict:
    """Per-frame forward predictive log-likelihoods, {chase_id: (s, values)}."""
    data, log_b, lengths = _scored(chases, fit, start, mp)
    ff = forward_frames(log_b, lengths, np.log(fit.pi), np.log(fit.A))
    return {a.chase_id: (s, ff[i, : lengths[i]]) for i, (a, s, *_) in enumerate(data)}


def score_switching_ar(
    chases: list, fit: ARFit, start: int | None = None, mp: FeatureParams = FeatureParams()
) -> dict:
    data, log_b, lengths = _scored(chases, fit, start, mp)
    _, _, ll = forward_backward_batch(log_b, lengths, np.log(fit.pi), np.log(fit.A))
    per = {a.chase_id: (float(lc), len(y)) for (a, _, y, *_), lc in zip(data, ll, strict=True)}
    tot = sum(v[0] for v in per.values())
    n = sum(v[1] for v in per.values())
    return {"loglik": tot, "n_frames": n, "per_frame": tot / max(n, 1), "per_chase": per}
