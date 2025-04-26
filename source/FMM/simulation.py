import math
import numpy as np
from quadtree import build_tree
import kernels
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Coulomb constant
k = 1.0

class Particle:
    """Particle in 2D with charge q."""
    def __init__(self, x, y, q):
        self.x = x
        self.y = y
        self.pos = (x, y)
        self.q = q
        self.phi = 0.0
        self.fx = 0.0
        self.fy = 0.0


def distance(p1, p2):
    """Distance between 2 points in 2D"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


def outer(tnode, p_order):
    """Compute multipole expansion recursively using kernels."""
    if tnode.is_leaf():
        tnode.outer = kernels.multipole(
            tnode.get_points(),
            center=tnode.center,
            p=p_order
        )
    else:
        tnode.outer = np.zeros(p_order+1, dtype=complex)
        for child in tnode:
            outer(child, p_order)
            z0 = complex(*child.center) - complex(*tnode.center)
            tnode.outer += kernels.M2M(child.outer, z0)


def inner(tnode):
    """Accumulate local expansions to leaves and evaluate."""
    # Shift parent's local expansion to this node
    z0 = complex(*tnode.parent.center) - complex(*tnode.center)
    tnode.inner = kernels.L2L(tnode.parent.inner, z0)
    # Add interactions from well-separated cells
    for tin in tnode.interaction_set:
        z0 = complex(*tin.center) - complex(*tnode.center)
        tnode.inner += kernels.M2L(tin.outer, z0)

    if tnode.is_leaf():
        zc = complex(*tnode.center)
        for p in tnode.get_points():
            z = complex(*p.pos)
            # potential via local expansion
            phi_loc = np.real(sum(
                tnode.inner[j] * (z - zc)**j
                for j in range(len(tnode.inner))
            ))
            p.phi += k * p.q * phi_loc
            # field = derivative of local expansion
            deriv = [j * tnode.inner[j] for j in range(1, len(tnode.inner))]
            E = np.polyval(deriv[::-1], z - zc)
            Ex, Ey = E.real, -E.imag
            # force = q * E (repulsive)
            p.fx += p.q * Ex
            p.fy += p.q * Ey

        # Direct interactions with near neighbors
        for nn in tnode.nearest_neighbors:
            force_naive(tnode.get_points(), nn.get_points())
            potential_naive(tnode.get_points(), nn.get_points())
        # Direct all-to-all inside this cell
        force_naive(tnode.get_points())
        potential_naive(tnode.get_points())
    else:
        for child in tnode:
            inner(child)


def potential(particles, tree_thresh=None, bbox=None, p_order=5):
    """Fast Multipole Method evaluation: resets and computes phi & forces."""
    for p in particles:
        p.phi = p.fx = p.fy = 0.0

    tree = build_tree(particles, tree_thresh, bbox=bbox)
    # Upward pass: compute multipole expansions
    outer(tree.root, p_order)
    # Downward pass: initialize root local expansion
    tree.root.inner = np.zeros(p_order+1, dtype=complex)
    for child in tree.root:
        inner(child)


def potential_naive(targets, sources=None):
    """
    Direct-sum Coulomb potential.

    If `sources` is None, does an all-to-all among `targets` (skipping self).
    Otherwise, does a cross-sum: every p in targets accumulates potential from every s in sources.
    Returns an array phi of length len(targets).
    """
    phi = np.zeros(len(targets), dtype=float)

    if sources is None:
        # all-to-all within the same list
        for i, p in enumerate(targets):
            for s in targets[:i] + targets[i+1:]:
                dx = p.x - s.x
                dy = p.y - s.y
                r = math.hypot(dx, dy)
                if r == 0:
                    continue
                p.phi += k * p.q * s.q * math.log(r)
            phi[i] = p.phi
    else:
        # cross interactions between two lists
        for i, p in enumerate(targets):
            for s in sources:
                dx = p.x - s.x
                dy = p.y - s.y
                r = math.hypot(dx, dy)
                if r == 0:
                    continue
                p.phi += k * p.q * s.q * math.log(r)
            phi[i] = p.phi

    return phi


def force_naive(targets, sources=None):
    """
    Direct-sum Coulomb forces.

    If `sources` is None, does an all-to-all among `targets` (skipping self).
    Otherwise, does a cross-sum: every p in targets feels every s in sources.
    """
    if sources is None:
        # all-to-all within the same list
        for i, p in enumerate(targets):
            for s in targets[:i] + targets[i+1:]:
                dx = p.x - s.x
                dy = p.y - s.y
                r2 = dx*dx + dy*dy
                if r2 == 0:
                    continue
                r = math.sqrt(r2)
                # Coulomb force magnitude = k * q_i * q_j / r
                f = k * p.q * s.q / r
                # project onto components: (dx, dy)/r
                p.fx += f * dx / r
                p.fy += f * dy / r
    else:
        # cross interactions between two lists
        for p in targets:
            for s in sources:
                dx = p.x - s.x
                dy = p.y - s.y
                r2 = dx*dx + dy*dy
                if r2 == 0:
                    continue
                r = math.sqrt(r2)
                f = k * p.q * s.q / r
                p.fx += f * dx / r
                p.fy += f * dy / r

# ----- Simulation and Animation Classes -----

class Body(Particle):
    def __init__(self, position, velocity, mass):
        super().__init__(position[0], position[1], mass)
        self.mass = mass
        self.v = np.array(velocity, dtype=float)

class Simulation:
    def __init__(self, bodies, dt=0.001, nterms=5):
        self.bodies = bodies
        self.dt = dt
        self.nterms = nterms

    def step(self):
        # Leapfrog integrator
        forces = self.compute_forces()
        for b, F in zip(self.bodies, forces):
            b.v += 0.5 * self.dt * F / b.mass
        for b in self.bodies:
            b.x += self.dt * b.v[0]
            b.y += self.dt * b.v[1]
            b.pos = (b.x, b.y)
        forces = self.compute_forces()
        for b, F in zip(self.bodies, forces):
            b.v += 0.5 * self.dt * F / b.mass

    def compute_forces(self):
        for b in self.bodies:
            b.phi = b.fx = b.fy = 0.0
        potential(self.bodies, tree_thresh=10, p_order=self.nterms)
        return [np.array((b.fx, b.fy), dtype=float) for b in self.bodies]

class Animation:
    def __init__(self, bodies, simulation, mass_scale=1000):
        self.bodies = bodies
        self.sim = simulation
        self.mass_scale = mass_scale

        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.axis('off')

        self.scat = self.ax.scatter(
            [b.x for b in bodies],
            [b.y for b in bodies],
            s=[b.mass * self.mass_scale for b in bodies],
            c='white',
            marker='o',
            alpha=1
        )
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_autoscale_on(False)

    def _update(self, i):
        self.sim.step()
        xs = [b.x for b in self.bodies]
        ys = [b.y for b in self.bodies]
        ss = [b.mass * self.mass_scale for b in self.bodies]
        self.scat.set_offsets(np.column_stack((xs, ys)))
        self.scat.set_sizes(ss)
        return (self.scat,)

    def show(self, frames=500, interval=30):
        self.ani = FuncAnimation(
            self.fig, self._update,
            frames=frames, interval=interval, blit=False
        )
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    bodies = [
        Body(
            position=np.random.uniform(-10, 10, 2),
            velocity=np.random.uniform(-2, 2, 2),
            mass=np.random.uniform(0.1, 1.0)
        )
        for _ in range(100)
    ]


    sim = Simulation(bodies, dt=0.005, nterms=4)
    anim = Animation(bodies, sim, mass_scale=1)
    anim.show()
