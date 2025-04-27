from itertools import chain
import numpy as np
from scipy.special import binom
from quadtree import build_tree


class Point:
    """Point in 2D"""
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.pos = (x, y)


class Particle(Point):
    """A Charged Particle"""
    def __init__(self, x, y, charge):
        super(Particle, self).__init__(x, y)
        self.q = charge
        self.phi = 0


def distance(p1, p2):
    """Distance between 2 points in 2D"""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) + 0.1


def multipole(particles, center=(0,0), nterms=5):
    """Compute a multipole expansion up to nterms terms"""
    coeffs = np.empty(nterms + 1, dtype=complex)
    coeffs[0] = sum(p.q for p in particles)
    coeffs[1:] = [
        sum(-p.q * complex(p.x - center[0], p.y - center[1])**k / k
            for p in particles)
        for k in range(1, nterms+1)
    ]
    return coeffs


def M2M(coeffs, z0):
    """Update multipole expansion coefficients for a center shift"""
    shifted = np.empty_like(coeffs)
    shifted[0] = coeffs[0]
    for l in range(1, len(coeffs)):
        shifted[l] = sum(
            coeffs[k] * z0**(l - k) * binom(l-1, k-1)
            for k in range(1, l)
        ) - coeffs[0] * z0**l / l
    return shifted


def outer(tnode, nterms):
    """Compute outer multipole expansion recursively"""
    if tnode.is_leaf():
        tnode.outer = multipole(tnode.get_points(),
                                center=tnode.center,
                                nterms=nterms)
    else:
        tnode.outer = np.zeros(nterms+1, dtype=complex)
        for child in tnode:
            outer(child, nterms)
            z0 = complex(*child.center) - complex(*tnode.center)
            tnode.outer += M2M(child.outer, z0)


def M2L(coeffs, z0):
    """Convert outer expansion to inner (local) about z0"""
    m = len(coeffs)
    inner = np.empty_like(coeffs)
    # zero‐th term
    inner[0] = (
        sum((coeffs[k] / z0**k) * (-1)**k for k in range(1, m))
        + coeffs[0] * np.log(-z0)
    )
    # higher terms
    for l in range(1, m):
        inner[l] = (
            sum((coeffs[k] / z0**k) * binom(l+k-1, k-1) * (-1)**k
                for k in range(1, m))
            / (z0**l)
            - coeffs[0] / (l * z0**l)
        )
    return inner


def L2L(coeffs, z0):
    """Shift inner (Taylor) expansion to new center"""
    m = len(coeffs)
    shifted = np.empty_like(coeffs)
    for l in range(m):
        shifted[l] = sum(
            coeffs[k] * binom(k, l) * (-z0)**(k - l)
            for k in range(l, m)
        )
    return shifted


def inner(tnode):
    """
    Downward pass: compute local expansions for every cell,
    then add far‐field via locals in leaves.
    """
    # ──────── downward pass ────────
    if tnode.parent is None:
        # root already has its .inner set externally
        pass
    else:
        # 1) shift parent local → this local
        z0 = complex(*tnode.center) - complex(*tnode.parent.center)
        tnode.inner = L2L(tnode.parent.inner, z0)

        # 2) add contributions from each interaction cell
        for tin in tnode.interaction_set():
            z0 = complex(*tnode.center) - complex(*tin.center)
            tnode.inner += M2L(tin.outer, z0)

    # ──────── evaluate at leaves ────────
    if tnode.is_leaf():
        # far‐field via local expansion
        zc = complex(*tnode.center)
        coeffs = tnode.inner
        for p in tnode.get_points():
            z = complex(*p.pos)
            p.phi -= np.real(np.polyval(coeffs[::-1], z - zc))

        # near‐field direct sums from neighbors
        for nn in tnode.nearest_neighbors:
            potentialDDS(tnode.get_points(), nn.get_points())

        # self‐leaf direct sum
        _ = potentialDS(tnode.get_points())
    else:
        for child in tnode:
            inner(child)


def potential(particles, bbox=None, tree_thresh=2, nterms=5, boundary='wall'):
    """FMM evaluation of all‐to‐all Coulomb potential"""
    tree = build_tree(particles, tree_thresh, bbox=bbox, boundary=boundary)
    outer(tree.root, nterms)

    # FIX: no far‐field outside the root — start with zero local expansion
    tree.root.inner = np.zeros_like(tree.root.outer)

    # downward pass
    any(inner(child) for child in tree.root)


def potentialFMM(tree, nterms=5):
    """FMM evaluation given a pre‐built quadtree"""
    outer(tree.root, nterms)

    tree.root.inner = np.zeros_like(tree.root.outer)

    any(inner(child) for child in tree.root)


def potentialDDS(particles, sources):
    """Direct sum from external sources"""
    for p in particles:
        for s in sources:
            r = distance(p.pos, s.pos)
            p.phi -= p.q * np.log(r)


def potentialDS(particles):
    """Direct sum among particles in same leaf"""
    for i, pi in enumerate(particles):
        for pj in particles[:i] + particles[i+1:]:
            r = distance(pi.pos, pj.pos)
            pi.phi -= pi.q * np.log(r)
    # return array if needed
    return np.array([p.phi for p in particles])