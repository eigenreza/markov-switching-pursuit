"""Hidden Markov machinery over the law index: forward-backward, Viterbi, and the
maximisation steps for plain and input-dependent (logistic) transitions.

The core routines work on a padded batch of chases at once, looping only over time, so that
an expectation step over hundreds of chases costs a few milliseconds. Scaled recursions in
linear space (Rabiner's scaling) are used, with the emission densities shifted per frame by
their maximum to avoid underflow; the shifts are added back to the log-likelihood.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def pad(arrays: list[np.ndarray], fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged (T_i, ...) arrays into (n, T_max, ...) with lengths."""
    lengths = np.array([len(a) for a in arrays], dtype=int)
    T = int(lengths.max()) if len(lengths) else 0
    out = np.full((len(arrays), T, *arrays[0].shape[1:]), fill, dtype=float)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out, lengths


def forward_backward_batch(log_b: np.ndarray, lengths: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray):
    """Posterior marginals gamma (n, T, K), pairwise posteriors xi (n, T-1, K, K), and the log
    marginal likelihood per chase (n,). ``log_A`` is (K, K) for fixed transitions or
    (n, T-1, K, K) for transitions that vary over time. Entries beyond each chase's length are
    zero (gamma, xi) and ignored."""
    n, T, K = log_b.shape
    tv = log_A.ndim == 4
    A = np.exp(log_A)
    shift = log_b.max(axis=2)
    b = np.exp(log_b - shift[:, :, None])
    pi = np.exp(log_pi)
    alpha = np.zeros((n, T, K))
    c = np.ones((n, T))
    a0 = pi[None, :] * b[:, 0]
    c[:, 0] = a0.sum(axis=1)
    alpha[:, 0] = a0 / c[:, 0, None]
    for t in range(1, T):
        active = t < lengths
        if not active.any():
            break
        pred = np.einsum("nj,njk->nk", alpha[:, t - 1], A[:, t - 1]) if tv else alpha[:, t - 1] @ A
        at = pred * b[:, t]
        ct = at.sum(axis=1)
        ct = np.where(ct > 0, ct, 1.0)
        alpha[:, t] = np.where(active[:, None], at / ct[:, None], 0.0)
        c[:, t] = np.where(active, ct, 1.0)
    valid = np.arange(T)[None, :] < lengths[:, None]
    loglik = np.sum(np.where(valid, np.log(c) + shift, 0.0), axis=1)
    beta = np.ones((n, T, K))
    for t in range(T - 2, -1, -1):
        active = (t + 1) < lengths
        nxt = b[:, t + 1] * beta[:, t + 1]
        bt = np.einsum("njk,nk->nj", A[:, t], nxt) if tv else nxt @ A.T
        bt = bt / c[:, t + 1, None]
        beta[:, t] = np.where(active[:, None], bt, 1.0)
    gamma = alpha * beta
    gamma = np.where(valid[:, :, None], gamma, 0.0)
    if T > 1:
        A_t = A if tv else A[None, None, :, :]
        xi = alpha[:, :-1, :, None] * A_t * (b[:, 1:] * beta[:, 1:])[:, :, None, :] / c[:, 1:, None, None]
        xi = np.where(valid[:, 1:, None, None], xi, 0.0)
    else:
        xi = np.zeros((n, 0, K, K))
    return gamma, xi, loglik


def filtered_batch(
    log_b: np.ndarray, lengths: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray
) -> np.ndarray:
    """Filtered state probabilities p(z_t | y_1..y_t) (n, T, K): the forward pass alone, which
    uses no observation after frame t. Zero beyond each chase's length."""
    n, T, K = log_b.shape
    tv = log_A.ndim == 4
    A = np.exp(log_A)
    shift = log_b.max(axis=2)
    b = np.exp(log_b - shift[:, :, None])
    pi = np.exp(log_pi)
    alpha = np.zeros((n, T, K))
    a0 = pi[None, :] * b[:, 0]
    alpha[:, 0] = a0 / a0.sum(axis=1)[:, None]
    for t in range(1, T):
        active = t < lengths
        if not active.any():
            break
        pred = np.einsum("nj,njk->nk", alpha[:, t - 1], A[:, t - 1]) if tv else alpha[:, t - 1] @ A
        at = pred * b[:, t]
        ct = at.sum(axis=1)
        ct = np.where(ct > 0, ct, 1.0)
        alpha[:, t] = np.where(active[:, None], at / ct[:, None], 0.0)
    return alpha


def forward_frames(
    log_b: np.ndarray, lengths: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray
) -> np.ndarray:
    """Per-frame predictive log-likelihoods log p(y_t | y_<t) (n, T), zero beyond each length.
    Their sum over a chase equals the marginal log-likelihood."""
    n, T, _ = log_b.shape
    tv = log_A.ndim == 4
    A = np.exp(log_A)
    shift = log_b.max(axis=2)
    b = np.exp(log_b - shift[:, :, None])
    pi = np.exp(log_pi)
    out = np.zeros((n, T))
    a0 = pi[None, :] * b[:, 0]
    c0 = a0.sum(axis=1)
    alpha = a0 / c0[:, None]
    out[:, 0] = np.log(c0) + shift[:, 0]
    for t in range(1, T):
        active = t < lengths
        if not active.any():
            break
        pred = np.einsum("nj,njk->nk", alpha, A[:, t - 1]) if tv else alpha @ A
        at = pred * b[:, t]
        ct = at.sum(axis=1)
        ct = np.where(ct > 0, ct, 1.0)
        alpha = np.where(active[:, None], at / ct[:, None], alpha)
        out[:, t] = np.where(active, np.log(ct) + shift[:, t], 0.0)
    return out


def viterbi_batch(
    log_b: np.ndarray, lengths: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray
) -> np.ndarray:
    """Most probable state paths (n, T); entries beyond each length are -1."""
    n, T, K = log_b.shape
    tv = log_A.ndim == 4
    score = np.full((n, T, K), -np.inf)
    back = np.zeros((n, T, K), dtype=int)
    score[:, 0] = log_pi[None, :] + log_b[:, 0]
    for t in range(1, T):
        A = log_A[:, t - 1] if tv else log_A[None, :, :]
        cand = score[:, t - 1, :, None] + A  # (n, K, K)
        back[:, t] = np.argmax(cand, axis=1)
        score[:, t] = log_b[:, t] + np.max(cand, axis=1)
    paths = np.full((n, T), -1, dtype=int)
    for i in range(n):
        L = int(lengths[i])
        paths[i, L - 1] = int(np.argmax(score[i, L - 1]))
        for t in range(L - 2, -1, -1):
            paths[i, t] = back[i, t + 1, paths[i, t + 1]]
    return paths


def forward_backward(log_b: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray):
    """Single-chase convenience wrapper: gamma (T, K), xi (T-1, K, K), loglik."""
    lA = log_A[None] if log_A.ndim == 3 else log_A
    g, xi, ll = forward_backward_batch(log_b[None], np.array([len(log_b)]), log_pi, lA)
    return g[0], xi[0], float(ll[0])


def viterbi(log_b: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray) -> np.ndarray:
    lA = log_A[None] if log_A.ndim == 3 else log_A
    return viterbi_batch(log_b[None], np.array([len(log_b)]), log_pi, lA)[0]


def mstep_plain(
    xi: np.ndarray, gamma: np.ndarray, lengths: np.ndarray, prior_stay: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form updates of the initial distribution and the transition matrix from pooled
    expected counts (padded entries are zero). A small pseudo-count on staying keeps rows away
    from zero."""
    K = gamma.shape[2]
    counts = np.full((K, K), 1e-3) + prior_stay * np.eye(K) + xi.sum(axis=(0, 1))
    A = counts / counts.sum(axis=1, keepdims=True)
    pi = np.full(K, 1e-3) + gamma[:, 0].sum(axis=0)
    return pi / pi.sum(), A


def transition_logits(W: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Log transition matrices (..., K, K) from weights W (K, K, P) and covariates Z (..., P).
    The diagonal (staying) is the reference with weights fixed at zero."""
    K = W.shape[0]
    logits = np.einsum("ijp,...p->...ij", W, Z)
    idx = np.arange(K)
    logits[..., idx, idx] = 0.0
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def mstep_input(xi: np.ndarray, Z: np.ndarray, W0: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    """Weighted multinomial logistic regression for the transition weights: maximise the expected
    complete-data log-likelihood of the transitions, with a small ridge penalty. ``xi`` is
    (n, T-1, K, K) and ``Z`` (n, T-1, P) holds the covariates of the frame transitioned into;
    padded entries of xi are zero and carry no weight."""
    K, _, P = W0.shape
    off = ~np.eye(K, dtype=bool)
    xi_all = xi.reshape(-1, K, K)
    Z_all = Z.reshape(-1, P)
    keep = xi_all.sum(axis=(1, 2)) > 0
    xi_all, Z_all = xi_all[keep], Z_all[keep]

    def unpack(w):
        W = np.zeros((K, K, P))
        W[off] = w.reshape(-1, P)
        return W

    def f(w):
        W = unpack(w)
        logA = transition_logits(W, Z_all)
        val = -np.sum(xi_all * logA) + 0.5 * l2 * np.sum(w**2)
        prob = np.exp(logA)
        row = xi_all.sum(axis=2, keepdims=True)
        resid = xi_all - row * prob
        grad = -np.einsum("nij,np->ijp", resid, Z_all)
        return val, grad[off].ravel() + l2 * w

    res = minimize(f, W0[off].ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return unpack(res.x)
