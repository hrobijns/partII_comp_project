import numpy as np

from quadtree import QuadTree, k, soft, dt, theta

# --- Body definition ---
class Body:
    def __init__(self, position, velocity, charge, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge   = charge
        self.mass     = mass
        self.force    = np.zeros(2)

# --- Simulation using BH forces + symplectic step ---
class Simulation:
    def __init__(self, bodies, space_size, theta):
        self.bodies     = bodies
        self.space_size = space_size
        self.theta      = theta
        self.compute_forces()

    def compute_forces(self):
        # build tree
        root = QuadTree(-self.space_size, self.space_size,
                        -self.space_size, self.space_size)
        for b in self.bodies:
            root.insert(b)
        # compute each body’s force
        for b in self.bodies:
            b.force = root.compute_force(b, self.theta)

    def step(self):
        # half-kick
        for b in self.bodies:
            b.velocity += 0.5*(b.force/b.mass)*dt
        # drift
        for b in self.bodies:
            b.position += b.velocity*dt
        # recompute forces at new pos
        self.compute_forces()
        # second half-kick
        for b in self.bodies:
            b.velocity += 0.5*(b.force/b.mass)*dt