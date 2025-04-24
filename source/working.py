import math
import numpy as np
from scipy.special import binom
from quadtree import build_tree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ——— Gravitational constant ———
G = 1.0


class Point:
    """Point in 2D"""
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.pos = (x, y)


class Particle(Point):
    """A 'massive' particle for FMM (we keep .q as the mass)"""
    def __init__(self, x, y, mass):
        super(Particle, self).__init__(x, y)
        self.q = mass           # used throughout as the mass
        self.phi = 0.0          # gravitational potential energy
        self.fx = 0.0           # force components
        self.fy = 0.0


def distance(p1, p2):
    """Distance between 2 points in 2D"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


def multipole(particles, center=(0,0), nterms=5):
    """Compute a multipole expansion up to nterms terms, scaled by G"""
    coeffs = np.empty(nterms + 1, dtype=complex)
    # monopole term = G * Σ m_i
    coeffs[0] = G * sum(p.q for p in particles)
    # higher moments
    coeffs[1:] = [
        G * sum(-p.q * complex(p.x - center[0], p.y - center[1])**k / k
                for p in particles)
        for k in range(1, nterms+1)
    ]
    return coeffs


def _shift_mpexp(coeffs, z0):
    """Update multipole expansion coefficients for a center shift"""
    shift = np.empty_like(coeffs)
    shift[0] = coeffs[0]
    shift[1:] = [
        sum(coeffs[k] * z0**(l - k) * binom(l-1, k-1)
            - (coeffs[0] * z0**l) / l
            for k in range(1, l))
        for l in range(1, len(coeffs))
    ]
    return shift


def _outer_mpexp(tnode, nterms):
    """Compute outer multipole expansion recursively"""
    if tnode.is_leaf():
        tnode.outer = multipole(tnode.get_points(),
                                 center=tnode.center, nterms=nterms)
    else:
        tnode.outer = np.zeros(nterms+1, dtype=complex)
        for child in tnode:
            _outer_mpexp(child, nterms)
            z0 = (complex(*child.center)
                  - complex(*tnode.center))
            tnode.outer += _shift_mpexp(child.outer, z0)


def _convert_oi(coeffs, z0):
    """Convert outer to inner expansion about z0"""
    inner = np.empty_like(coeffs)
    inner[0] = (
        sum((coeffs[k]/z0**k) * (-1)**k
            for k in range(1, len(coeffs)))
        + coeffs[0] * np.log(-z0)
    )
    inner[1:] = [
        (1/z0**l) * sum((coeffs[k]/z0**k) * binom(l+k-1, k-1) * (-1)**k
                        for k in range(1, len(coeffs)))
        - coeffs[0] / (z0**l * l)
        for l in range(1, len(coeffs))
    ]
    return inner


def _shift_texp(coeffs, z0):
    """Shift inner (Taylor) expansions to new center"""
    return [
        sum(coeffs[k] * binom(k, l) * (-z0)**(k-l)
            for k in range(l, len(coeffs)))
        for l in range(len(coeffs))
    ]


def _inner(tnode):
    """Accumulate multipole + direct interactions to leaves"""
    # pull down the parent inner
    z0 = complex(*tnode.parent.center) - complex(*tnode.center)
    tnode.inner = _shift_texp(tnode.parent.inner, z0)

    # add contributions from well-separated cells
    for tin in tnode.interaction_set:
        z0 = complex(*tin.center) - complex(*tnode.center)
        tnode.inner = [
            ti + oi
            for ti, oi in zip(tnode.inner,
                              _convert_oi(tin.outer, z0))
        ]

    if tnode.is_leaf():
        zc = complex(*tnode.center)
        # evaluate local expansion
        for p in tnode.get_points():
            z = complex(*p.pos)
            # potential (log-kernel)
            p.phi -= np.real(np.polyval(tnode.inner[::-1], z - zc))
            # field = derivative of that potential
            deriv = [l * tnode.inner[l] for l in range(1, len(tnode.inner))]
            E = np.polyval(deriv[::-1], z - zc)
            Ex, Ey = E.real, -E.imag
            # force = mass * field, and we want it attractive,
            # but the local expansion already carries G, so just:
            p.fx += p.q * Ex
            p.fy += p.q * Ey

        # Direct interactions with neighbor cells
        for nn in tnode.nearest_neighbors:
            pts = tnode.get_points()
            srcs = nn.get_points()
            if srcs:
                forceDDS(pts, srcs)
                potentialDDS(pts, srcs)

        # Direct all-to-all inside this cell
        forceDS(tnode.get_points())
        _ = potentialDS(tnode.get_points())
    else:
        for child in tnode:
            _inner(child)


def potential(particles, bbox=None, tree_thresh=None,
              nterms=5, boundary='wall'):
    """Fast Multipole Method evaluation: resets and computes phi & forces"""
    for p in particles:
        p.phi = p.fx = p.fy = 0.0

    tree = build_tree(particles, tree_thresh,
                      bbox=bbox, boundary=boundary)
    _outer_mpexp(tree.root, nterms)
    tree.root.inner = np.zeros(nterms+1, dtype=complex)
    any(_inner(child) for child in tree.root)


def potentialDDS(particles, sources):
    """Direct sum of gravitational potential from separate sources"""
    for p in particles:
        for s in sources:
            r = distance(p.pos, s.pos)
            if r == 0:
                continue
            # U = -G m1 m2 log(r)
            p.phi -= G * p.q * s.q * math.log(r)


def forceDDS(particles, sources):
    """Direct sum of gravitational forces from separate sources"""
    for p in particles:
        for s in sources:
            dx = p.x - s.x
            dy = p.y - s.y
            r2 = dx*dx + dy*dy
            if r2 == 0:
                continue
            # magnitude G m1 m2 / r^2
            f = G * p.q * s.q / r2
            # subtract so that force is attractive
            p.fx -= f * dx
            p.fy -= f * dy


def potentialDS(particles):
    """Direct sum of gravitational potential all-to-all"""
    phi = np.zeros(len(particles))
    for i, p in enumerate(particles):
        for s in particles[:i] + particles[i+1:]:
            r = distance(p.pos, s.pos)
            if r == 0:
                continue
            p.phi -= G * p.q * s.q * math.log(r)
        phi[i] = p.phi
    return phi


def forceDS(particles):
    """Direct sum of gravitational forces all-to-all"""
    for i, p in enumerate(particles):
        for s in particles[:i] + particles[i+1:]:
            dx = p.x - s.x
            dy = p.y - s.y
            r2 = dx*dx + dy*dy
            if r2 == 0:
                continue
            f = G * p.q * s.q / r2
            p.fx -= f * dx
            p.fy -= f * dy


# ----- Simulation and Animation Classes, unchanged except for naming----

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
        potential(particles=self.bodies,
                  tree_thresh=10,
                  nterms=self.nterms)
        return [np.array((b.fx, b.fy), dtype=float)
                for b in self.bodies]

class Animation:
    def __init__(self, bodies, simulation, mass_scale=1000):
        self.bodies     = bodies
        self.sim        = simulation
        self.mass_scale = mass_scale

        # --- set up black background ---
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')      # figure bg
        self.ax.set_facecolor('black')             # axes bg
        self.ax.axis('off')                        # no axes, ticks, or grid

        # --- white “star” markers ---
        self.scat = self.ax.scatter(
            [b.x for b in bodies],
            [b.y for b in bodies],
            s=[b.mass * self.mass_scale for b in bodies],
            c='white',     # white bodies
            marker='o',
            alpha=1
        )

        # keep same limits (you can adjust)
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
            mass=np.random.uniform(0.1, 1.0),
        )
        for _ in range(1000)
    ]
    central_mass = 1e3
    black_hole = Body(
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        mass=central_mass
    )
    bodies.append(black_hole)

    sim  = Simulation(bodies, dt=0.005, nterms=4)
    anim = Animation(bodies, sim, mass_scale=1)
    anim.show()