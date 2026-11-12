"""Stable LPV-DS (linear parameter-varying dynamical system), numpy/scipy only.

After Figueroa & Billard (2018), *A Physically-Consistent Bayesian Non-Parametric Mixture
Model for Dynamical System Learning*:

    ẋ = f(x) = Σ_k γ_k(x) A_k (x − x*),     γ_k(x) = posterior of a GMM on positions

Global asymptotic stability at x* follows from a quadratic Lyapunov function
V = (x − x*)ᵀ P (x − x*) if A_kᵀP + P A_k ≺ 0 for every k. Instead of solving the SDP of the
original paper, the constraint is built into the parameterization:

    P = Q Qᵀ + εI (trace normalized),   B_k = −(L_k L_kᵀ + εI) + (W_k − W_kᵀ),   A_k = P⁻¹ B_k
    ⇒ A_kᵀP + P A_k = B_kᵀ + B_k = −2(L_k L_kᵀ + εI) ≺ 0

so any parameter value is stable and the fit is an unconstrained nonlinear least squares on
the demonstrated velocities (``scipy.optimize.least_squares``).

Clustering: the GMM is fitted on [x, w·ẋ/‖ẋ‖] (position plus scaled velocity direction), a
lightweight stand-in for the physically-consistent GMM: points close in space but moving in
different directions fall in different components. γ_k(x) uses the position marginals only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from so101_tpgmm.tpgmm.ds import fit_gmm
from so101_tpgmm.tpgmm.gmm import GMM, gaussian_logpdf, logsumexp

EPS = 1e-3


def _tril(v: np.ndarray, d: int) -> np.ndarray:
    M = np.zeros((d, d))
    M[np.tril_indices(d)] = v
    return M


def _skew(v: np.ndarray, d: int) -> np.ndarray:
    M = np.zeros((d, d))
    M[np.triu_indices(d, 1)] = v
    return M - M.T


@dataclass
class LPVDS:
    """Fitted LPV-DS in some task frame (positions in metres, velocities in m/s)."""

    gmm: GMM  # scheduling GMM on positions (d-dimensional)
    A: np.ndarray  # (K, d, d)
    P: np.ndarray  # (d, d) Lyapunov matrix
    target: np.ndarray  # (d,) attractor

    @property
    def dim(self) -> int:
        return int(self.target.shape[0])

    def gamma(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        L = np.stack([np.log(self.gmm.priors[k] + 1e-300) + gaussian_logpdf(X, self.gmm.means[k], self.gmm.covs[k])
                      for k in range(self.gmm.n_components)], axis=1)
        return np.exp(L - logsumexp(L, axis=1)[:, None])

    def velocity(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        G = self.gamma(X)
        return np.einsum("nk,kij,nj->ni", G, self.A, X - self.target)

    def lyapunov(self, X: np.ndarray) -> np.ndarray:
        D = np.atleast_2d(X) - self.target
        return np.einsum("ni,ij,nj->n", D, self.P, D)

    def to_dict(self) -> dict:
        return {"priors": self.gmm.priors.tolist(), "means": self.gmm.means.tolist(), "covs": self.gmm.covs.tolist(),
                "A": self.A.tolist(), "P": self.P.tolist(), "target": self.target.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "LPVDS":
        return cls(GMM(np.array(d["priors"]), np.array(d["means"]), np.array(d["covs"])), np.array(d["A"]),
                   np.array(d["P"]), np.array(d["target"]))


def _unpack(theta: np.ndarray, K: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    nq, nl, nw = d * (d + 1) // 2, d * (d + 1) // 2, d * (d - 1) // 2
    Q = _tril(theta[:nq], d)
    P = Q @ Q.T + EPS * np.eye(d)
    P = P * d / np.trace(P)
    Pinv = np.linalg.inv(P)
    A = np.zeros((K, d, d))
    off = nq
    for k in range(K):
        Lk = _tril(theta[off:off + nl], d)
        Wk = _skew(theta[off + nl:off + nl + nw], d)
        off += nl + nw
        A[k] = Pinv @ (-(Lk @ Lk.T + EPS * np.eye(d)) + Wk)
    return A, P


def fit_lpvds(pos: np.ndarray, vel: np.ndarray, target: np.ndarray, k_range=(3, 4, 5, 6, 8),
              direction_weight: float = 0.05, max_points: int = 2000, max_nfev: int = 80,
              seed: int = 0) -> tuple[LPVDS, dict]:
    """Fit a stable LPV-DS to (position, velocity) samples.

    Args:
        pos, vel: (N, d) demonstrations in the task frame.
        target: (d,) attractor.
        k_range: number of scheduling components, chosen by BIC of the clustering GMM.
        direction_weight: scale [m] of the unit velocity direction in the clustering features.
        max_points: subsample for the least-squares fit (speed).
    """
    pos, vel = np.asarray(pos, float), np.asarray(vel, float)
    N, d = pos.shape
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    feat = np.c_[pos, direction_weight * vel / np.maximum(speed, 1e-6)]
    reg = np.r_[np.full(d, 1e-6), np.full(d, 1e-4 * direction_weight**2)]
    best, bic = None, {}
    for K in k_range:
        g, ll = fit_gmm(feat, K, reg, seed=seed)
        D = 2 * d
        bic[K] = -2 * ll + ((K - 1) + K * (D + D * (D + 1) // 2)) * np.log(N)
        if best is None or bic[K] < bic[best[0]]:
            best = (K, g)
    K, g = best
    sched = GMM(g.priors.copy(), g.means[:, :d].copy(), g.covs[:, :d, :d].copy())
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, min(N, max_points), replace=False)
    X, V = pos[idx], vel[idx]
    tmp = LPVDS(sched, np.zeros((K, d, d)), np.eye(d), np.asarray(target, float))
    G = tmp.gamma(X)
    Dx = X - target

    def residual(theta):
        A, _ = _unpack(theta, K, d)
        return (np.einsum("nk,kij,nj->ni", G, A, Dx) - V).ravel()

    nq, nl, nw = d * (d + 1) // 2, d * (d + 1) // 2, d * (d - 1) // 2
    theta0 = np.zeros(nq + K * (nl + nw))
    theta0[:nq] = np.eye(d)[np.tril_indices(d)]
    for k in range(K):  # start from A_k = -I (L_k = I)
        theta0[nq + k * (nl + nw): nq + k * (nl + nw) + nl] = np.eye(d)[np.tril_indices(d)]
    sol = least_squares(residual, theta0, method="trf", max_nfev=max_nfev)
    A, P = _unpack(sol.x, K, d)
    model = LPVDS(sched, A, P, np.asarray(target, float))
    rmse = float(np.sqrt(np.mean(np.sum((model.velocity(pos) - vel) ** 2, axis=1))))
    return model, {"K": K, "bic": bic, "velocity_rmse": rmse, "nfev": int(sol.nfev)}


@dataclass
class ScalarDS:
    """Stable 1-D DS scheduled by position: ż = Σ_k γ_k(x) exp(a_k) (z* − z)."""

    log_gain: np.ndarray  # (K,)
    target: float

    def rate(self, gamma: np.ndarray, z: np.ndarray | float) -> np.ndarray:
        return (gamma @ np.exp(self.log_gain)) * (self.target - np.asarray(z))

    @classmethod
    def fit(cls, gamma: np.ndarray, z: np.ndarray, zdot: np.ndarray, target: float) -> "ScalarDS":
        e = target - z

        def res(a):
            return (gamma @ np.exp(a)) * e - zdot

        sol = least_squares(res, np.zeros(gamma.shape[1]), bounds=(-6, 4))
        return cls(sol.x, float(target))
