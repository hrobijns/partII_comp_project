import numpy as np
from quadtree import QuadTree
from kernels import multipole, M2M, M2L, L2L

class Point:
    """Point in 2D"""
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.position = (x, y)

class Particle(Point):
    """A Charged Particle"""
    def __init__(self, x, y, charge):
        super().__init__(x, y)
        self.q = charge
        self.phi = 0.0


def distance(p1, p2):
    """Distance between 2 points in 2D (with softening)."""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) + 1e-2


def outer(tnode, nterms):
    """Upward pass: build multipole expansions on every cell."""
    if tnode.is_leaf():
        tnode.outer = multipole(
            tnode.get_points(),
            center=tnode.center,
            nterms=nterms
        )
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

    # 2) at leaves, evaluate far-field (via local) and near-field (direct)
    if tnode.is_leaf():
        zc = complex(*tnode.center)
        coeffs = tnode.inner
        for p in tnode.get_points():
            z = complex(*p.position)
            p.phi -= np.real(np.polyval(coeffs[::-1], z - zc))
        for nn in tnode.neighbors:
            potentialDDS(tnode.get_points(), nn.get_points())
        potentialDS(tnode.get_points())
    else:
        for child in tnode:
            inner(child)


def potential(particles, tree_thresh=2, nterms=5):
    """
    Build a quadtree over `particles`, run FMM, and return potentials
    in the same order as the input list.
    """
    xs = [p.x for p in particles]
    ys = [p.y for p in particles]
    bnd = (min(xs), min(ys), max(xs), max(ys))
    tree = QuadTree(particles, boundary=bnd, max_points=tree_thresh)

    # upward + downward passes
    outer(tree.root, nterms)
    tree.root.inner = np.zeros_like(tree.root.outer)
    inner(tree.root)

    return np.array([p.phi for p in particles])


def potentialDDS(particles, sources):
    """Direct sum: external sources → targets."""
    for p in particles:
        for s in sources:
            r = distance(p.position, s.position)
            p.phi -= s.q * np.log(r)


def potentialDS(particles):
    """Direct sum among particles in the same leaf."""
    for i, pi in enumerate(particles):
        for pj in particles[:i] + particles[i+1:]:
            r = distance(pi.position, pj.position)
            pi.phi -= pj.q * np.log(r)
    return np.array([p.phi for p in particles])

