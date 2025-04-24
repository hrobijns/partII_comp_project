'''
Implementation of the 2D Fast Multipole Method for a Coulomb potential
with force calculation for simulation/animation
'''

__author__ = 'Luis Barroso-Luque'

import math
import numpy as np
from scipy.special import binom
from quadtree import build_tree


class Point:
    """Point in 2D"""
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.pos = (x, y)


class Particle(Point):
    """A Charged Particle with potential and force attributes"""
    def __init__(self, x, y, charge):
        super(Particle, self).__init__(x, y)
        self.q = charge
        self.phi = 0.0
        self.fx = 0.0
        self.fy = 0.0


def distance(p1, p2):
    """Distance between 2 points in 2D"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


def multipole(particles, center=(0,0), nterms=5):
    """Compute a multipole expansion up to nterms terms"""
    coeffs = np.empty(nterms + 1, dtype=complex)
    coeffs[0] = sum(p.q for p in particles)
    coeffs[1:] = [sum(-p.q * complex(p.x - center[0], p.y - center[1])**k / k
                      for p in particles)
                  for k in range(1, nterms+1)]
    return coeffs


def _shift_mpexp(coeffs, z0):
    """Update multipole expansion coefficients for a center shift"""
    shift = np.empty_like(coeffs)
    shift[0] = coeffs[0]
    shift[1:] = [sum(coeffs[k] * z0**(l - k) * binom(l-1, k-1)
                      - (coeffs[0] * z0**l) / l
                      for k in range(1, l))
                  for l in range(1, len(coeffs))]
    return shift


def _outer_mpexp(tnode, nterms):
    """Compute outer multipole expansion recursively"""
    if tnode.is_leaf():
        tnode.outer = multipole(tnode.get_points(), center=tnode.center, nterms=nterms)
    else:
        tnode.outer = np.zeros(nterms+1, dtype=complex)
        for child in tnode:
            _outer_mpexp(child, nterms)
            z0 = complex(*child.center) - complex(*tnode.center)
            tnode.outer += _shift_mpexp(child.outer, z0)


def _convert_oi(coeffs, z0):
    """Convert outer to inner expansion about z0"""
    inner = np.empty_like(coeffs)
    inner[0] = (sum((coeffs[k]/z0**k) * (-1)**k
                    for k in range(1, len(coeffs)))
                + coeffs[0] * np.log(-z0))
    inner[1:] = [(1/z0**l) * sum((coeffs[k]/z0**k) * binom(l+k-1, k-1) * (-1)**k
                                  for k in range(1, len(coeffs)))
                 - coeffs[0] / (z0**l * l)
                 for l in range(1, len(coeffs))]
    return inner


def _shift_texp(coeffs, z0):
    """Shift inner (Taylor) expansions to new center"""
    return [sum(coeffs[k] * binom(k, l) * (-z0)**(k-l)
                for k in range(l, len(coeffs)))
            for l in range(len(coeffs))]


def _inner(tnode):
    """Compute inner expansions and accumulate potential + forces for leaf particles"""
    # shift parent's inner to this cell
    z0 = complex(*tnode.parent.center) - complex(*tnode.center)
    tnode.inner = _shift_texp(tnode.parent.inner, z0)
    # add contributions from interaction list
    for tin in tnode.interaction_set:
        z0 = complex(*tin.center) - complex(*tnode.center)
        tnode.inner = [ti + oi for ti, oi in zip(tnode.inner, _convert_oi(tin.outer, z0))]

    if tnode.is_leaf():
        # far-field via local expansion derivative
        zc = complex(*tnode.center)
        for p in tnode.get_points():
            z = complex(*p.pos)
            # potential
            p.phi -= np.real(np.polyval(tnode.inner[::-1], z-zc))
            # compute field E_complex = dW/dz
            deriv = [l * tnode.inner[l] for l in range(1, len(tnode.inner))]
            E = np.polyval(deriv[::-1], z-zc)
            Ex, Ey = E.real, -E.imag
            p.fx += p.q * Ex
            p.fy += p.q * Ey
        # near-field direct forces & potential
        for nn in tnode.nearest_neighbors:
            forceDDS(tnode.get_points(), nn.get_points())
            potentialDDS(tnode.get_points(), nn.get_points())
        # self-cell direct all-to-all
        forceDS(tnode.get_points())
        _ = potentialDS(tnode.get_points())
    else:
        for child in tnode:
            _inner(child)


def potential(particles, bbox=None, tree_thresh=None, nterms=5, boundary='wall'):
    """Fast Multipole Method evaluation: resets and computes phi for each particle"""
    # reset
    for p in particles:
        p.phi = p.fx = p.fy = 0.0
    tree = build_tree(particles, tree_thresh, bbox=bbox, boundary=boundary)
    _outer_mpexp(tree.root, nterms)
    tree.root.inner = np.zeros(nterms+1, dtype=complex)
    any(_inner(child) for child in tree.root)


def potentialFMM(tree, nterms=5):
    """Like above but using a prebuilt tree"""
    for p in tree.root.get_points():
        p.phi = p.fx = p.fy = 0.0
    _outer_mpexp(tree.root, nterms)
    tree.root.inner = np.zeros(nterms+1, dtype=complex)
    any(_inner(child) for child in tree.root)


def potentialDDS(particles, sources):
    """Direct sum of potential from separate sources"""
    for p in particles:
        for s in sources:
            r = distance(p.pos, s.pos)
            p.phi -= p.q * s.q * math.log(r)


def forceDDS(particles, sources):
    """Direct sum of forces from separate sources"""
    for p in particles:
        for s in sources:
            dx = p.x - s.x
            dy = p.y - s.y
            r2 = dx*dx + dy*dy
            if r2 == 0:
                continue
            f = p.q * s.q / r2
            p.fx += f * dx
            p.fy += f * dy


def potentialDS(particles):
    """Direct sum of potential all-to-all"""
    phi = np.zeros(len(particles))
    for i, p in enumerate(particles):
        for s in particles[:i] + particles[i+1:]:
            r = distance(p.pos, s.pos)
            p.phi -= p.q * s.q * math.log(r)
        phi[i] = p.phi
    return phi


def forceDS(particles):
    """Direct sum of forces all-to-all"""
    for i, p in enumerate(particles):
        for s in particles[:i] + particles[i+1:]:
            dx = p.x - s.x
            dy = p.y - s.y
            r2 = dx*dx + dy*dy
            if r2 == 0:
                continue
            f = p.q * s.q / r2
            p.fx += f * dx
            p.fy += f * dy
    # forces stored in p.fx, p.fy for each particle
