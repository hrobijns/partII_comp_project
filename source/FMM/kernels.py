import numpy as np
from scipy.special import binom


def multipole(particles, center=(0.0, 0.0), p=5):
    """
    Compute multipole expansion (moments) up to order p for the 2D logarithmic kernel.
    Particles must have attributes x, y, q (charge/mass).
    center: (cx, cy)
    Returns M: np.ndarray of length p+1 of complex coefficients:
    M[k] = \sum_i q_i (z_i - z0)^k.
    """
    z0 = complex(*center)
    M = np.zeros(p+1, dtype=complex)
    # Monopole
    M[0] = sum(p.q for p in particles)
    # Higher moments
    for k in range(1, p+1):
        M[k] = sum(p.q * (complex(p.x, p.y) - z0)**k for p in particles)
    return M


def M2M(M_child, c):
    """
    Translate multipole M_child by vector c (complex):
    M_parent[k] = \sum_{n=0}^k binom(k, n) * M_child[n] * c^(k-n).
    c = z_child_center - z_parent_center.
    """
    p = len(M_child) - 1
    M_parent = np.zeros_like(M_child)
    for k in range(p+1):
        for n in range(k+1):
            M_parent[k] += binom(k, n) * M_child[n] * c**(k-n)
    return M_parent


def M2L(M, z0):
    """
    Convert source multipole M at z_source to local expansion L at z_target.
    z0 = z_source_center - z_target_center (complex).
    Returns L: np.ndarray of length p+1 with:
      L[0] = M[0]*log(z0) + \sum_{j=1}^p (-1)^j * M[j] / (j * z0^j)
      L[k] = -M[0]/(k * z0^k) + \sum_{j=1}^p (-1)^j * M[j] * binom(j+k-1, k) / z0^(j+k), for k>=1.
    """
    p = len(M) - 1
    L = np.zeros(p+1, dtype=complex)
    # L0 term
    L[0] = M[0] * np.log(z0)
    for j in range(1, p+1):
        L[0] += (-1)**j * M[j] / (j * z0**j)
    # Lk terms
    for k in range(1, p+1):
        L[k] = -M[0] / (k * z0**k)
        for j in range(1, p+1):
            L[k] += (-1)**j * M[j] * binom(j+k-1, k) / (z0**(j+k))
    return L


def L2L(L_parent, c):
    """
    Shift local expansion L_parent by vector c (complex) to get child local expansion:
    c = z_parent_center - z_child_center.
    L_child[k] = \sum_{n=k}^p binom(n, k) * L_parent[n] * c^(n-k).
    """
    p = len(L_parent) - 1
    L_child = np.zeros_like(L_parent)
    for k in range(p+1):
        for n in range(k, p+1):
            L_child[k] += binom(n, k) * L_parent[n] * c**(n-k)
    return L_child
