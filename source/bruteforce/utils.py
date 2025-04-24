import numpy as np
from typing import List
from .core import Body

def total_energy(bodies: List[Body]) -> float:
    """Compute total (kinetic + potential) energy."""
    kin = sum(0.5 * b.mass * np.dot(b.velocity, b.velocity) for b in bodies)
    pot = 0.0
    for i, b1 in enumerate(bodies):
        for b2 in bodies[i+1:]:
            r = np.linalg.norm(b1.position - b2.position)
            pot -= g * b1.mass * b2.mass / r
    return kin + pot


def total_momentum(bodies: List[Body]) -> np.ndarray:
    """Compute vector sum of momenta."""
    return sum(b.mass * b.velocity for b in bodies)