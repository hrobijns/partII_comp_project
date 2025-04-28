import numpy as np
from scipy.special import binom

def multipole(particles, center=(0, 0), nterms=5):
    """Compute multipole expansion coefficients up to order 'nterms' around 'center'."""
    coeffs = np.empty(nterms + 1, dtype=complex)
    # Monopole: total charge
    coeffs[0] = sum(p.q for p in particles)
    # Higher-order terms
    for k in range(1, nterms + 1):
        coeffs[k] = sum(
            -p.q * complex(p.x - center[0], p.y - center[1])**k / k
            for p in particles
        )
    return coeffs


def M2M(coeffs, z0):
    """Shift multipole expansion coefficients by vector 'z0'."""
    shifted = np.empty_like(coeffs)
    shifted[0] = coeffs[0]
    for l in range(1, len(coeffs)):
        shifted[l] = (
            sum(
                coeffs[k] * z0**(l - k) * binom(l-1, k-1)
                for k in range(1, l)
            )
            - coeffs[0] * z0**l / l
        )
    return shifted


def M2L(coeffs, z0):
    """Convert multipole (outer) expansion to a local (inner) expansion about `z0`."""
    m = len(coeffs)
    inner = np.empty_like(coeffs)
    # Constant term
    inner[0] = (
        sum((coeffs[k] / z0**k) * (-1)**k for k in range(1, m))
        + coeffs[0] * np.log(-z0)
    )
    # Higher-order terms
    for l in range(1, m):
        inner[l] = (
            sum(
                (coeffs[k] / z0**k) * binom(l + k - 1, k - 1) * (-1)**k
                for k in range(1, m)
            ) / (z0**l)
            - coeffs[0] / (l * z0**l)
        )
    return inner


def L2L(coeffs, z0):
    """Shift a local (inner) expansion to a new center by `z0`."""
    m = len(coeffs)
    shifted = np.empty_like(coeffs)
    for l in range(m):
        shifted[l] = sum(
            coeffs[k] * binom(k, l) * (-z0)**(k - l)
            for k in range(l, m)
        )
    return shifted
