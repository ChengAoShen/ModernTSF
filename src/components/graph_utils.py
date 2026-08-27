"""Graph adjacency normalization utilities for spatiotemporal models.

Ported from CauAir's ``src/utils/graph_algo.py``. These functions convert a raw
adjacency matrix (numpy) into various normalized forms used by GNN-based models.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

import torch


def normalize_adj_mx(
    adj_mx: np.ndarray, adj_type: str, return_type: str = "dense"
) -> list[np.ndarray]:
    """Normalize an adjacency matrix using the specified method.

    Parameters
    ----------
    adj_mx : np.ndarray
        Raw adjacency matrix of shape ``(N, N)``.
    adj_type : str
        One of: ``normlap``, ``scalap``, ``symadj``, ``transition``,
        ``doubletransition``, ``identity``, ``origin``.
    return_type : str
        ``dense`` (default) or ``coo``.

    Returns
    -------
    list[np.ndarray]
        List of normalized adjacency matrices (1 or 2 depending on type).
    """
    if adj_type == "normlap":
        adj = [_normalized_laplacian(adj_mx)]
    elif adj_type == "scalap":
        adj = [_scaled_laplacian(adj_mx)]
    elif adj_type == "symadj":
        adj = [_sym_adj(adj_mx)]
    elif adj_type == "transition":
        adj = [_asym_adj(adj_mx)]
    elif adj_type == "doubletransition":
        adj = [_asym_adj(adj_mx), _asym_adj(np.transpose(adj_mx))]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "origin":
        adj_mx = adj_mx.copy()
        np.fill_diagonal(adj_mx, 1)
        adj = [adj_mx.astype(np.float32)]
    else:
        return []

    if return_type == "dense":
        adj = [np.asarray(a.astype(np.float32).todense()) if sp.issparse(a) else a.astype(np.float32) for a in adj]
    elif return_type == "coo":
        adj = [a.tocoo() if sp.issparse(a) else sp.coo_matrix(a) for a in adj]
    return adj


def adj_to_supports(
    adj_mx: np.ndarray,
    adj_type: str = "doubletransition",
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """Convert a raw adjacency matrix to a list of support tensors.

    This is the standard pipeline used by GWNet, DCRNN, etc.
    """
    normalized = normalize_adj_mx(adj_mx, adj_type, return_type="dense")
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in normalized]


def cheb_poly(L: np.ndarray, Ks: int) -> np.ndarray:
    """Compute Chebyshev polynomials up to order Ks.

    Parameters
    ----------
    L : np.ndarray
        Scaled Laplacian of shape ``(N, N)``.
    Ks : int
        Number of Chebyshev polynomials.

    Returns
    -------
    np.ndarray
        Shape ``(Ks, N, N)``.
    """
    n = L.shape[0]
    LL = [np.eye(n), L.copy()]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[i - 1]) - LL[i - 2])
    return np.asarray(LL)


# --- Internal helpers ---


def _normalized_laplacian(adj_mx: np.ndarray) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def _scaled_laplacian(
    adj_mx: np.ndarray, lambda_max: float | None = None, undirected: bool = True
) -> sp.csr_matrix:
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = _normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max_arr, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max_arr[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    return (2 / lambda_max * L) - I


def _sym_adj(adj_mx: np.ndarray) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


def _asym_adj(adj_mx: np.ndarray) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj)
