"""Autonomous dynamical system (DS) learned from the same demonstrations: GMR-DS.

The TP-GMM of this package is *time-indexed* (Gaussians over [t, x], regression on t), so it
does not define a vector field by itself. A DS is obtained by fitting a GMM on
[position, velocity] in a task frame and conditioning on the position (Khansari-Zadeh &
Billard's GMR-DS, the unconstrained form of SEDS):

    f(x) = E[ẋ | x] = Σ_k h_k(x) (μ_ẋ,k + Σ_ẋx,k Σ_xx,k⁻¹ (x − μ_x,k))

Without SEDS's constraints the field is **not** guaranteed to converge to the target: the
plots show where it does (negative divergence, streamlines ending at the cube) and where the
data end (low mixture density, where GMR extrapolates arbitrarily).
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2

from so101_tpgmm.tpgmm.gmm import GMM, gaussian_logpdf, gmr, logsumexp


def fit_gmm(X: np.ndarray, K: int, reg: np.ndarray, max_iter: int = 300, tol: float = 1e-6, seed: int = 0) -> tuple[GMM, float]:
    """Plain EM for a full-covariance GMM, k-means initialization. Returns (GMM, log-likelihood)."""
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    scale = X.std(axis=0) + 1e-12
    _, lab = kmeans2(X / scale, K, minit="++", seed=seed)
    pri, mu, S = np.zeros(K), np.zeros((K, D)), np.zeros((K, D, D))
    for k in range(K):
        Xk = X[lab == k] if np.sum(lab == k) > D else X
        pri[k], mu[k], S[k] = max(np.mean(lab == k), 1e-3), Xk.mean(0), np.cov(Xk.T) + np.diag(reg)
    pri /= pri.sum()
    prev = -np.inf
    for _ in range(max_iter):
        L = np.stack([np.log(pri[k]) + gaussian_logpdf(X, mu[k], S[k]) for k in range(K)], axis=1)
        norm = logsumexp(L, axis=1)
        G = np.exp(L - norm[:, None])
        Nk = G.sum(0) + 1e-12
        pri = Nk / N
        for k in range(K):
            w = G[:, k] / Nk[k]
            mu[k] = w @ X
            d = X - mu[k]
            S[k] = (d * w[:, None]).T @ d + np.diag(reg)
        ll = float(norm.sum())
        if abs(ll - prev) <= tol * abs(ll):
            break
        prev = ll
    return GMM(pri, mu, S), ll


def fit_ds(pos: np.ndarray, vel: np.ndarray, k_range=(4, 6, 8, 10), pos_reg: float = 1e-6,
           vel_reg: float = 1e-6) -> tuple[GMM, dict[int, float]]:
    """GMM on [x, ẋ] with K by BIC. ``pos``/``vel`` are (N, d)."""
    X = np.c_[pos, vel]
    d = pos.shape[1]
    reg = np.r_[np.full(d, pos_reg), np.full(d, vel_reg)]
    D = X.shape[1]
    fits, bic = {}, {}
    for K in k_range:
        g, ll = fit_gmm(X, K, reg)
        fits[K] = g
        bic[K] = -2 * ll + ((K - 1) + K * (D + D * (D + 1) // 2)) * np.log(len(X))
    return fits[min(bic, key=bic.get)], bic


def field(g: GMM, P: np.ndarray) -> np.ndarray:
    """Velocity field f(x) at points ``P`` (N, d)."""
    d = P.shape[1]
    v, _ = gmr(g, P, list(range(d)), list(range(d, 2 * d)))
    return v


def density(g: GMM, P: np.ndarray) -> np.ndarray:
    """Marginal mixture density of the positions (to mask regions without data)."""
    d = P.shape[1]
    idx = list(range(d))
    L = np.stack([np.log(g.priors[k]) + gaussian_logpdf(P, g.means[k, idx], g.covs[k][np.ix_(idx, idx)])
                  for k in range(g.n_components)], axis=1)
    return np.exp(logsumexp(L, axis=1))


def divergence(U: np.ndarray, V: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """∂u/∂x + ∂v/∂y on a regular grid (arrays indexed [iy, ix])."""
    return np.gradient(U, xs, axis=1) + np.gradient(V, ys, axis=0)


def integrate(g: GMM, x0: np.ndarray, dt: float = 0.02, T: float = 8.0) -> np.ndarray:
    """Euler integration of the learned field from ``x0``."""
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(int(T / dt)):
        xs.append(xs[-1] + dt * field(g, xs[-1][None])[0])
    return np.array(xs)
