"""Gaussian mixture utilities: log-densities, Gaussian products and Gaussian mixture regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GMM:
    """A Gaussian mixture in one coordinate system."""

    priors: np.ndarray  # (K,)
    means: np.ndarray  # (K, D)
    covs: np.ndarray  # (K, D, D)

    @property
    def n_components(self) -> int:
        return int(self.priors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])


def gaussian_logpdf(X: np.ndarray, mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """log N(x | mu, S) for every row of ``X`` (Cholesky-based, numerically stable)."""
    X = np.atleast_2d(X)
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (X - mu).T)
    maha = np.sum(z * z, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (maha + logdet + X.shape[1] * np.log(2 * np.pi))


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def gaussian_product(means: list[np.ndarray], covs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Product of Gaussians N(mu_j, S_j): S = (sum S_j^-1)^-1, mu = S sum S_j^-1 mu_j."""
    precs = [np.linalg.inv(S) for S in covs]
    S = np.linalg.inv(np.sum(precs, axis=0))
    mu = S @ np.sum([P @ m for P, m in zip(precs, means)], axis=0)
    return mu, 0.5 * (S + S.T)


def gmr(gmm: GMM, x_in: np.ndarray, in_idx: list[int], out_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian mixture regression E[x_out | x_in] and Cov[x_out | x_in].

    Args:
        gmm: joint mixture.
        x_in: (N, I) inputs.
        in_idx, out_idx: dimension indices.

    Returns:
        (N, O) conditional means and (N, O, O) conditional covariances.
    """
    x_in = np.asarray(x_in, dtype=np.float64).reshape(-1, len(in_idx))
    N, K, O = x_in.shape[0], gmm.n_components, len(out_idx)
    ii, oo = np.ix_(in_idx, in_idx), np.ix_(out_idx, in_idx)
    logw = np.zeros((N, K))
    mu_c = np.zeros((K, N, O))
    S_c = np.zeros((K, O, O))
    for k in range(K):
        Sii = gmm.covs[k][ii]
        logw[:, k] = np.log(gmm.priors[k] + 1e-300) + gaussian_logpdf(x_in, gmm.means[k, in_idx], Sii)
        Soi = gmm.covs[k][oo]
        gain = Soi @ np.linalg.inv(Sii)
        mu_c[k] = gmm.means[k, out_idx] + (x_in - gmm.means[k, in_idx]) @ gain.T
        S_c[k] = gmm.covs[k][np.ix_(out_idx, out_idx)] - gain @ Soi.T
    h = np.exp(logw - logsumexp(logw, axis=1)[:, None])  # (N, K)
    mean = np.einsum("nk,kno->no", h, mu_c)
    second = np.einsum("nk,koq->noq", h, S_c) + np.einsum("nk,kno,knq->noq", h, mu_c, mu_c)
    cov = second - np.einsum("no,nq->noq", mean, mean)
    return mean, 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
