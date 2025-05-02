import numpy as np
from scipy.special import binom

def multipole(particles, center=(0.0, 0.0), nterms=5):
    """
    Build the multipole moments M[0..nterms] around `center`.
    """
    # extract charges and complex positions relative to center
    q = np.array([p.q for p in particles], dtype=complex)                # (P,)
    z = np.array([p.x - center[0] + 1j*(p.y - center[1])
                  for p in particles], dtype=complex)                  # (P,)

    # orders 1...nterms
    ks = np.arange(1, nterms+1)                                          # (K,)
    # z**k for each particle and each k: shape (P, K)
    Zp = z[:, None] ** ks[None, :]
    
    M = np.empty(nterms+1, dtype=complex)
    M[0] = q.sum()
    # for k>=1:  M[k] = - sum_p q_p * z_p**k / k
    M[1:] = - (q[:, None] * Zp).sum(axis=0) / ks
    return M

def M2M(child_coeffs, z0):
    """
    Promote a child multipole series `child_coeffs` by translating its center by z0,
    returning the parent-series of the same length.
    """
    c = np.asarray(child_coeffs, dtype=complex)
    n = c.size
    # indices 1..n-1 for the sums
    k = np.arange(1, n)
    l = np.arange(1, n)
    # form a (ℓ, k) grid
    Lj, Kj = np.meshgrid(l, k, indexing='ij')      # both shape (n-1,n-1)
    # binomial and shift kernels (zero out k>ℓ if necessary)
    B = binom(Lj-1, Kj-1) * (Kj <= Lj)
    Z = z0**(Lj - Kj)
    # sum over k for each ℓ
    term = (c[k][None, :] * B * Z).sum(axis=1)     # shape (n-1,)
    # subtract log-shift part
    shift = c[0] * (z0**l) / l
    M_parent = np.empty_like(c, dtype=complex)
    M_parent[0] = c[0]
    M_parent[1:] = term - shift
    return M_parent

def M2L(child_coeffs, z0):
    """
    Convert a multipole series at one center into a local series at another,
    separated by z0.
    """
    c = np.asarray(child_coeffs, dtype=complex)
    n = c.size

    L = np.empty_like(c, dtype=complex)
    ks = np.arange(1, n)

    # L[0] = c0*log(z0) + sum_{k>=1} c[k] / z0^k
    L[0] = c[0]*np.log(z0) + (c[ks] / z0**ks).sum()

    # for ℓ>=1:
    #   s[ℓ] = sum_{k>=1} c[k] * binom(k+ℓ-1,ℓ) / z0^(k+ℓ)
    #   L[ℓ] = (-1)^ℓ * ( s[ℓ] - c[0]/(ℓ * z0^ℓ) )
    ell = np.arange(1, n)
    # build (ℓ, k) grid
    Ell, Kk = np.meshgrid(ell, ks, indexing='ij')  # both (n-1,n-1)
    B = binom(Kk + Ell - 1, Ell)
    Z = z0**(-(Kk + Ell))
    s = (c[ks][None, :] * B * Z).sum(axis=1)   # shape (n-1,)

    L[1:] = (-1)**ell * (s - c[0] / (ell * z0**ell))
    return L

def L2L(parent_locals, z0):
    """
    Translate a local expansion `parent_locals` by z0 to get new locals.
    """
    c = np.asarray(parent_locals, dtype=complex)
    n = c.size

    ell = np.arange(n)
    k   = np.arange(n)
    Ell, Kk = np.meshgrid(ell, k, indexing='ij')  # both (n,n)
    mask = (Kk >= Ell)
    B = binom(Kk, Ell) * mask
    Z = z0**(Kk - Ell) * mask

    # sum over k for each ℓ
    shifted = (c[None, :] * B * Z).sum(axis=1)   # shape (n,)
    return shifted