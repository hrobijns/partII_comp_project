import numpy as np
from scipy.special import binom

def multipole(particles, center=(0, 0), nterms=5):
    """Compute multipole expansion coefficients up to order 'nterms' around 'center'."""
    coeffs = np.empty(nterms + 1, dtype=complex)
    if len(particles) == 0:
        coeffs[:] = 0+0j
        return coeffs

    # monopole
    coeffs[0] = sum(p.q for p in particles)

    # higher moments
    for k in range(1, nterms+1):
        s = 0+0j
        for p in particles:
            z = complex(p.x - center[0], p.y - center[1])
            s += -p.q * z**k / k
        coeffs[k] = s

    return coeffs


def M2M(coeffs, z0):
    """Shift multipole expansion coefficients by vector 'z0'."""
    nmax = len(coeffs) - 1
    shift = np.empty_like(coeffs)
    shift[0] = coeffs[0]
    # for each order n ≥ 1:
    #   shift[n] = sum_{k=1..n} binom(n-1,k-1) coeffs[k] * (-z0)^(n-k)
    #              − coeffs[0] * (-z0)^n / n
    for n in range(1, nmax+1):
        s = sum(
            coeffs[k] * binom(n-1, k-1) * (-z0)**(n-k)
            for k in range(1, n+1)
        )
        shift[n] = s - coeffs[0] * (-z0)**n / n
    return shift


def M2L(coeffs, z0):
    """Convert multipole (outer) expansion to a local (inner) expansion about 'z0'."""
    inner = np.empty_like(coeffs)
    inner[0] = (sum([(coeffs[k]/z0**k)*(-1)**k for k in range(1, len(coeffs))]) +
          coeffs[0]*np.log(-z0))
    inner[1:] = [(1/z0**l)*sum([(coeffs[k]/z0**k)*binom(l+k-1, k-1)*(-1)**k
                 for k in range(1, len(coeffs))]) - coeffs[0]/((z0**l)*l)
                 for l in range(1, len(coeffs))]
    return inner


def L2L(coeffs, z0):
    """Shift a local (inner) expansion to a new center by 'z0'."""
    n = len(coeffs)
    shift = np.empty(n, dtype=complex)
    for l in range(n):
        s = 0+0j
        for k in range(l, n):
            s += coeffs[k] * binom(k, l) * (-z0)**(k-l)
        shift[l] = s
    return shift