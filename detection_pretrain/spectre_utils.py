

import numpy as np
import scipy as sp
import scipy.linalg
from scipy.linalg import sqrtm, solve
from scipy.sparse.linalg import aslinearoperator, LinearOperator, eigsh
from functools import partial
from dataclasses import dataclass


def cov_Tail(T, d, epsilon, tau):
    if T <= 10 * np.log(1 / epsilon):
        return 1
    return 3 * epsilon / (T * np.log(T)) ** 2


def Q(P):
    return 2 * np.linalg.norm(P) ** 2


def k_lowest_ind(A, k):
    assert k >= 0
    result = np.zeros(A.shape, np.bool_)
    result[np.argpartition(A, k)[:k]] = True
    return result


def krtv(A, B, v):
    """Khatri-Rao product times vector: (A ⊙ B)v."""
    (m, n), (p, q) = A.shape, B.shape
    assert n == q
    assert len(v) == n
    return (B @ (v * A.T) if n <= p else (B * v.T) @ A.T).ravel()


def tkrtv(A, B, v):
    """Transpose Khatri-Rao product times vector: (A ⊙ᵀ B)v."""
    (m, n), (p, q) = A.shape, B.shape
    assert m == p
    assert len(v) == n * q
    V = v.reshape((q, n))
    return np.sum((B @ V) * A if n <= q else B * (A @ V), axis=1)


def bisect_left_with_key(lst, value, key=None):
    # https://stackoverflow.com/a/42147515
    if key is None:
        key = lambda x: x

    def bis(lo, hi=len(lst)):
        while lo < hi:
            mid = (lo + hi) // 2
            if key(lst[mid]) < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return bis(0)


@dataclass
class CovEstimationFilterResult:
    filter: np.array = None
    sigma: np.array = None


def cov_estimation_filter(S_prime, epsilon, tau=0.1, *, limit=None):
    n, d = S_prime.shape
    C, C_prime = 10, 0
    Sigma_prime = S_prime.T @ S_prime / n

    # TODO: This can be optimized more
    Y = solve(sqrtm(Sigma_prime), S_prime.T, sym_pos=True).T
    x_inv_sqrt_Sigma_prime_x = np.einsum("ij,ji->i", Y, Y.T)

    def limit_mask(mask, scores):
        return CovEstimationFilterResult(
            filter=mask
            if limit is None
            else mask | k_lowest_ind(scores, max(0, n - limit))
        )

    mask = x_inv_sqrt_Sigma_prime_x >= C * d * np.log(n / tau)
    if mask.any():
        print("early filter")
        return limit_mask(~mask, x_inv_sqrt_Sigma_prime_x)

    Z = LinearOperator(
        (d * d, n), matvec=partial(krtv, Y.T, Y.T), rmatvec=partial(tkrtv, Y, Y)
    )
    Id_flat = aslinearoperator(np.eye(d).ravel()).T
    TS_prime = -Id_flat * Id_flat.T + (1 / n) * Z * Z.T
    lam, v = eigsh(TS_prime, 1)
    V = (v.reshape((d, d)) + v.reshape((d, d)).T) / 2
    if lam <= (1 + C * epsilon * np.log(1 / epsilon) ** 2) * Q(V) / 2:
        return CovEstimationFilterResult(sigma=Sigma_prime)

    ps = (np.sum(Y @ V * Y, axis=1) - np.trace(V))/np.sqrt(2)
    mu = np.median(ps)
    diffs = np.abs(ps - mu)

    # Can we vectorize this loop or make it fast somehow?
    for (i, diff) in enumerate(np.sort(diffs)):
        T = diff - 3
        if T <= C_prime:
            continue
        # print(i / n, cov_Tail(T, d, epsilon, tau))
        if 1 - (i / n) >= cov_Tail(T, d, epsilon, tau):
            return limit_mask(diffs <= T, diffs)

    return CovEstimationFilterResult()


def cov_estimation_iterate(
    S_prime, epsilon, tau=0.1, k=None, *, iters=None, limit=None, progress=True
):
    n = len(S_prime)
    i, orig_limit = 0, limit
    idxs = np.arange(n)
    while True:
        if iters is not None and i >= iters:
            break

        if k is None:
            S_prime_k = S_prime
        else:
            raise NotImplementedError

        result = cov_estimation_filter(S_prime_k, epsilon, tau, limit=limit)
        if result.sigma is not None:
            print(f"Terminating early {i} success...")
            break

        elif result.filter is not None:
            if limit is not None:
                limit -= len(result.filter) - np.sum(result.filter)
                assert limit >= 0

            S_prime = S_prime[result.filter]
            idxs = idxs[result.filter]

        else:
            print(f"Terminating early {i} fail...")
            break

        i += 1
        if limit == 0:
            break

    select = np.zeros(n, np.bool_)
    select[idxs] = True
    return select

def rcov(S_prime, epsilon, tau=0.1, *, iters=None, limit=None):
    select = cov_estimation_iterate(S_prime, epsilon, tau, iters=iters, limit=limit)
    selected = S_prime[select]
    return (selected.T @ selected) / len(S_prime)