import numpy as np
from quadtree import QuadTree
from kernels import multipole, M2M, M2L, L2L

def distance(p1, p2):
    """Distance between two 2D points (with small softening)."""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) 

class Point:
    """Point in 2D"""
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.position = (x, y)

class Particle(Point):
    """A charged particle with charge q and potential phi."""
    def __init__(self, x, y, charge):
        super().__init__(x, y)
        self.q = charge
        self.phi = 0.0

class FMM2D:
    """Fast Multipole Method in 2D using a quadtree and complex expansions."""
    def __init__(self, particles, max_points=2, nterms=5):
        self.particles = particles
        self.nterms = nterms
        xs = [p.x for p in particles]
        ys = [p.y for p in particles]
        boundary = (min(xs), min(ys), max(xs), max(ys))
        self.tree = QuadTree(self.particles, boundary=boundary, max_per_leaf=max_points)

    def upward_pass(self):
        """Compute multipole expansions (outer) from leaves up to the root."""
        def recurse(node):
            if node.is_leaf():
                # use node.points instead of get_points()
                node.outer = multipole(node.points, center=node.center, nterms=self.nterms)
            else:
                for child in node.children:
                    recurse(child)
                node.outer = np.zeros(self.nterms+1, dtype=complex)
                for child in node.children:
                    z0 = complex(*child.center) - complex(*node.center)
                    node.outer += M2M(child.outer, z0)
        recurse(self.tree.root)

    def downward_pass(self):
        """Compute local expansions (inner) from root down to leaves and evaluate at leaves."""
        root = self.tree.root
        root.inner = np.zeros(self.nterms+1, dtype=complex)
        
        def recurse(node):
            print(f"Level {node.level}: neighbors={len(node.neighbors)}, IL={len(node.interaction_list)}")
            if node.parent is not None:
                # shift parent's local expansion
                z0 = complex(*node.center) - complex(*node.parent.center)
                node.inner = L2L(node.parent.inner, z0)
                # add contributions from well-separated cells
                for in_node in node.interaction_list:
                    z1 = complex(*node.center) - complex(*in_node.center)
                    node.inner += M2L(in_node.outer, z1)
            if node.is_leaf():
                self.evaluate(node)
            else:
                for child in node.children:
                    recurse(child)
        recurse(root)

    def evaluate(self, node):
        """Evaluate potentials for particles in a leaf node."""
        zc = complex(*node.center)
        for p in node.points:
            z = complex(*p.position)
            # far-field via local expansion
            phi = 0.0
            for l, coeff in enumerate(node.inner):
                phi += coeff * (z - zc)**l
            # near-field via direct computation
            for neighbor in node.neighbors + [node]:
                for s in neighbor.points:
                    if s is not p:
                        r = distance(p.position, s.position)
                        phi -= s.q * np.log(r)
            p.phi = phi.real if np.iscomplexobj(phi) else phi

    def compute(self):
        """Perform FMM and return potentials in input order."""
        # reset potentials
        for p in self.particles:
            p.phi = 0.0
        # upward and downward passes
        self.upward_pass()
        self.downward_pass()
        return np.array([p.phi for p in self.particles])

def potential(particles, tree_thresh=2, nterms=5):
    """
    Convenience function: build FMM2D and compute potentials.
    :param particles: list of Particle objects
    :param tree_thresh: max bodies per leaf
    :param nterms: number of expansion terms
    """
    fmm = FMM2D(particles, max_points=tree_thresh, nterms=nterms)
    return fmm.compute()

def potentialDDS(particles, sources):
    """Direct sum: external sources → targets."""
    for p in particles:
        for s in sources:
            r = distance(p.position, s.position)
            p.phi -= s.q * np.log(r)
    return np.array([p.phi for p in particles])

def potentialDS(particles):
    """Direct sum among particles in the same leaf."""
    for i, pi in enumerate(particles):
        for pj in particles[:i] + particles[i+1:]:
            r = distance(pi.position, pj.position)
            pi.phi -= pj.q * np.log(r)
    return np.array([p.phi for p in particles])
