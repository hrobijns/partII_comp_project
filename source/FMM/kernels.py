import numpy as np
from scipy.special import binom

def multipole(particles, center=(0,0), p=5):
    """Compute a multipole expansion up to nterms terms"""

    coeffs = np.empty(p + 1, dtype=complex)
    coeffs[0] = sum(p.q for p in particles)
    coeffs[1:] = [sum([-p.q*complex(p.x - center[0], p.y - center[1])**k/k
                       for p in particles]) for k in range(1, p+1)]

    return coeffs


def M2M(coeffs, z0):
    """Update multipole expansion coefficients for a center shift"""
    shift = np.empty_like(coeffs)
    shift[0] = coeffs[0]
    shift[1:] = [sum([coeffs[k]*z0**(l - k)*binom(l-1, k-1) - (coeffs[0]*z0**l)/l
                      for k in range(1, l)]) for l in range(1, len(coeffs))]

    return shift


def M2L(coeffs, z0):
    """Convert outer to inner expansion about z0"""

    inner = np.empty_like(coeffs)
    inner[0] = (sum([(coeffs[k]/z0**k)*(-1)**k for k in range(1, len(coeffs))]) +
                coeffs[0]*np.log(-z0))
    inner[1:] = [(1/z0**l)*sum([(coeffs[k]/z0**k)*binom(l+k-1, k-1)*(-1)**k
                                for k in range(1, len(coeffs))]) - coeffs[0]/((z0**l)*l)
                                for l in range(1, len(coeffs))]
    return inner


def L2L(coeffs, z0):
    """Shift inner expansions (Taylor) to new center"""
    shift = np.empty_like(coeffs)
    shift = [sum([coeffs[k]*binom(k,l)*(-z0)**(k-l)
                  for k in range(l, len(coeffs))])
                  for l in range(len(coeffs))]
    return shift