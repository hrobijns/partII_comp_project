import numpy as np
from typing import List, Optional

# Gravitational constant in AU^3 M_sun^-1 day^-2
g = 2.959122082855911e-4

class Body:
    """
    Represents a celestial body with position, velocity, mass, and net force.
    """
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float
    ):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(2, dtype=float)

class Simulation:
    """
    Handles N-body integration using a symplectic leapfrog method.
    """
    def __init__(
        self,
        bodies: List[Body],
        dt: float = 1/24,
        softening: float = 1e-1
    ):
        self.bodies = bodies
        self.dt = dt
        self.softening = softening
        # initialize forces
        self.compute_forces()

    def compute_forces(self) -> None:
        """Compute pairwise gravitational forces with softening."""
        for b in self.bodies:
            b.force.fill(0.0)
        for i, b1 in enumerate(self.bodies):
            for j, b2 in enumerate(self.bodies):
                if i >= j:
                    continue
                diff = b2.position - b1.position
                dist2 = np.dot(diff, diff) + self.softening**2
                inv_dist = 1.0 / np.sqrt(dist2)
                f = g * b1.mass * b2.mass * inv_dist**3 * diff
                b1.force += f
                b2.force -= f  # Newton's third law

    def step(self) -> None:
        """Advance simulation by one time-step (leapfrog)."""
        # half-kick
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * self.dt
        # drift
        for b in self.bodies:
            b.position += b.velocity * self.dt
        # recompute forces
        self.compute_forces()
        # half-kick
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * self.dt

    def run(self, steps: int) -> None:
        """Run the simulation for a given number of steps."""
        for _ in range(steps):
            self.step()