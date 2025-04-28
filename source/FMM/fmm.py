import numpy as np
from scipy.special import binom
from quadtree import QuadTree

class Point:
    """Point in 2D"""
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.position = (x, y)

class Particle(Point):
    """A Charged Particle"""
    def __init__(self, x, y, charge):
        super().__init__(x, y)
        self.q   = charge
        self.phi = 0.0

def distance(p1, p2):
    """Distance between 2 points in 2D (with softening)."""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) + 1e-2

def multipole(particles, center=(0,0), nterms=5):
    """Compute a multipole expansion up to 'nterms' terms."""
    coeffs = np.empty(nterms + 1, dtype=complex)
    coeffs[0] = sum(p.q for p in particles)
    coeffs[1:] = [
        sum(-p.q * complex(p.x - center[0], p.y - center[1])**k / k
            for p in particles)
        for k in range(1, nterms+1)
    ]
    return coeffs

def M2M(coeffs, z0):
    """Shift multipole expansion coefficients by vector z0."""
    shifted = np.empty_like(coeffs)
    shifted[0] = coeffs[0]
    for l in range(1, len(coeffs)):
        shifted[l] = (
            sum(coeffs[k] * z0**(l - k) * binom(l-1, k-1)
                for k in range(1, l))
            - coeffs[0] * z0**l / l
        )
    return shifted

def M2L(coeffs, z0):
    """Convert multipole ('outer') expansion to a local ('inner') expansion about z0."""
    m = len(coeffs)
    inner = np.empty_like(coeffs)
    inner[0] = (
        sum((coeffs[k] / z0**k) * (-1)**k for k in range(1, m))
        + coeffs[0] * np.log(-z0)
    )
    for l in range(1, m):
        inner[l] = (
            sum((coeffs[k] / z0**k) * binom(l+k-1, k-1) * (-1)**k
                for k in range(1, m))
            / (z0**l)
            - coeffs[0] / (l * z0**l)
        )
    return inner

def L2L(coeffs, z0):
    """Shift a local ('inner') expansion to a new center by z0."""
    m = len(coeffs)
    shifted = np.empty_like(coeffs)
    for l in range(m):
        shifted[l] = sum(
            coeffs[k] * binom(k, l) * (-z0)**(k - l)
            for k in range(l, m)
        )
    return shifted

def outer(tnode, nterms):
    """Upward pass: build multipole expansions on every cell."""
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

def inner(tnode):
    """Downward pass + evaluation at leaves."""
    # 1) propagate local expansions
    if tnode.parent is not None:
        z0 = complex(*tnode.center) - complex(*tnode.parent.center)
        tnode.inner = L2L(tnode.parent.inner, z0)
        for tin in tnode.interaction_list:
            z1 = complex(*tnode.center) - complex(*tin.center)
            tnode.inner += M2L(tin.outer, z1)
    # 2) at leaves, evaluate both far‐field (via local) and near‐field (direct)
    if tnode.is_leaf():
        # far‐field via local expansion
        zc = complex(*tnode.center)
        coeffs = tnode.inner
        for p in tnode.get_points():
            z = complex(*p.position)
            p.phi -= np.real(np.polyval(coeffs[::-1], z - zc))
        # near‐field direct sums from neighbors
        for nn in tnode.neighbors:
            potentialDDS(tnode.get_points(), nn.get_points())
        # self‐leaf direct sum
        potentialDS(tnode.get_points())
    else:
        for child in tnode:
            inner(child)

def potential(particles, tree_thresh=2, nterms=5):
    """
    Build a quadtree over 'particles', run FMM, and return potential for each particle
    in the same order as the input list.
    """
    xs = [p.x for p in particles]
    ys = [p.y for p in particles]
    bnd = (min(xs), min(ys), max(xs), max(ys))
    tree = QuadTree(particles, boundary=bnd, max_points=tree_thresh)

    # upward + downward passes
    outer(tree.root,   nterms)
    tree.root.inner = np.zeros_like(tree.root.outer)
    inner(tree.root)

    # collect results
    return np.array([p.phi for p in particles])

def potentialDDS(particles, sources):
    """Direct sum: external sources → targets."""
    for p in particles:
        for s in sources:
            r = distance(p.position, s.position)
            p.phi -= s.q * np.log(r)

def potentialDS(particles):
    """Direct sum among particles in the *same* leaf."""
    for i, pi in enumerate(particles):
        for pj in particles[:i] + particles[i+1:]:
            r = distance(pi.position, pj.position)
            pi.phi -= pj.q * np.log(r)
    # (optional) return the array of φ if you need it immediately
    return np.array([p.phi for p in particles])
