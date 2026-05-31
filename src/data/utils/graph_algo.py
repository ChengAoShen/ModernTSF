"""Graph adjacency normalization utilities.

Ported verbatim from CauAir (src/utils/graph_algo.py). Provides the standard
normalization schemes used by spatiotemporal graph models.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def normalize_adj_mx(
    adj_mx: np.ndarray,
    adj_type: str,
    return_type: str = "dense",
) -> list[np.ndarray]:
    """Normalize an adjacency matrix according to *adj_type*.

    Parameters
    ----------
    adj_mx : np.ndarray
        Raw (N, N) adjacency matrix.
    adj_type : str
        One of ``"normlap"``, ``"scalap"``, ``"symadj"``, ``"transition"``,
        ``"doubletransition"``, ``"identity"``, ``"origin"``.
    return_type : str
        ``"dense"`` (default) returns dense numpy arrays; ``"coo"`` returns
        scipy COO sparse matrices.

    Returns
    -------
    list[np.ndarray]
        List of normalized adjacency matrices (most types return one;
        ``"doubletransition"`` returns two: forward and backward).
    """
    if adj_type == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == "symadj":
        adj = [calculate_sym_adj(adj_mx)]
    elif adj_type == "transition":
        adj = [calculate_asym_adj(adj_mx)]
    elif adj_type == "doubletransition":
        adj = [calculate_asym_adj(adj_mx), calculate_asym_adj(np.transpose(adj_mx))]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "origin":
        adj_mx = adj_mx.copy()
        np.fill_diagonal(adj_mx, 1)
        adj = [adj_mx.astype(np.float32)]
    else:
        return []

    if return_type == "dense":
        adj = [
            np.asarray(a.todense()).astype(np.float32) if sp.issparse(a) else a.astype(np.float32)
            for a in adj
        ]
    elif return_type == "coo":
        adj = [sp.coo_matrix(a) for a in adj]
    return adj


def calculate_normalized_laplacian(adj_mx: np.ndarray):
    """I - D^{-1/2} A D^{-1/2} (sparse)."""
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()


def calculate_scaled_laplacian(
    adj_mx: np.ndarray,
    lambda_max: float | None = None,
    undirected: bool = True,
):
    """(2/λ_max) L - I (sparse). Used for Chebyshev polynomial approximation."""
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    return (2 / lambda_max * L) - I


def calculate_sym_adj(adj_mx: np.ndarray):
    """D^{-1/2} A D^{-1/2} (sparse)."""
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)


def calculate_asym_adj(adj_mx: np.ndarray):
    """D^{-1} A (sparse). Row-stochastic transition matrix."""
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj_mx)


def calculate_cheb_poly(L: np.ndarray, Ks: int) -> np.ndarray:
    """Compute Chebyshev polynomial basis up to order *Ks*.

    Parameters
    ----------
    L : np.ndarray
        Scaled Laplacian (N, N).
    Ks : int
        Number of Chebyshev polynomials (order).

    Returns
    -------
    np.ndarray
        Shape ``(Ks, N, N)`` — the Chebyshev basis matrices.
    """
    n = L.shape[0]
    if sp.issparse(L):
        L = np.asarray(L.todense())
    LL = [np.eye(n), L.copy()]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[i - 1]) - LL[i - 2])
    return np.asarray(LL)


def load_adj_from_numpy(path: str) -> np.ndarray:
    """Load adjacency from a ``.npy`` file and fill diagonal with 1.

    Parameters
    ----------
    path : str
        Path to the ``.npy`` file containing an (N, N) adjacency matrix.

    Returns
    -------
    np.ndarray
        Adjacency matrix with self-loops (diagonal = 1).
    """
    adj_mx = np.load(path).astype(np.float32)
    np.fill_diagonal(adj_mx, 1.0)
    return adj_mx
