import numpy as np
from scipy.special import binom

def multipole(particles, center=(0,0), nterms=5):
    
    M = np.zeros(nterms+1, dtype=complex)
    M[0] = sum(p.q for p in particles)
    for k in range(1, nterms+1):
        M[k] = -sum(p.q * (complex(p.x - center[0], p.y - center[1])**k) / k
                    for p in particles)
    return M

def M2M(coeffs, z0):

    n = len(coeffs)
    M_parent = np.zeros_like(coeffs, dtype=complex)
    # Zeroth term (total charge) is invariant
    M_parent[0] = coeffs[0]

    # Higher terms
    for l in range(1, n):
        total = 0+0j
        # sum k = 1..ℓ of M_child[k] * binom(ℓ-1, k-1) * z0^(ℓ-k)
        for k in range(1, l+1):
            total += coeffs[k] * binom(l-1, k-1) * (z0**(l-k))
        # subtract the log‐shift contribution
        total -= coeffs[0] * (z0**l) / l
        M_parent[l] = total

    return M_parent

def M2L(coeffs, z0):

    n = len(coeffs)
    L = np.empty(n, dtype=complex)

    # 0th term
    L[0] = coeffs[0] * np.log(z0) \
           + sum(coeffs[k] / z0**k for k in range(1, n))

    # higher orders
    for ell in range(1, n):
        # sum over k=1..n-1 of M_k * binom(k+ell-1, ell) / z0^(k+ell)
        s = sum(
            coeffs[k] * binom(k + ell - 1, ell) / z0**(k + ell)
            for k in range(1, n)
        )
        # combine with monopole, multiply overall (-1)^ell
        L[ell] = (-1)**ell * (s - coeffs[0] / (ell * z0**ell))

    return L

def L2L(coeffs, z0):

    n = len(coeffs)
    shifted = np.empty(n, dtype=complex)

    for ell in range(n):
        s = 0+0j
        # sum k=ell..n-1 of coeffs[k] * binom(k, ell) * z0^(k-ell)
        for k in range(ell, n):
            s += coeffs[k] * binom(k, ell) * (z0)**(k-ell)
        shifted[ell] = s

    return shifted

