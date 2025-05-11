import numpy as np

k = 1.0      # Coulomb constant (in appropriate simulation units)
soft = 0.1   # softening length
dt = 0.005    # time step 

class Body:
    """
    Represents a charged particle with position, velocity, charge, mass, and accumulated force.

    Attributes:
        position (np.ndarray): 2D position vector of the body.
        velocity (np.ndarray): 2D velocity vector of the body.
        charge (float): Electric charge of the body.
        mass (float): Mass of the body (default = 1.0).
        force (np.ndarray): Accumulated force vector acting on the body.
    """

    def __init__(self, position, velocity, charge, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge = charge
        self.mass = mass
        self.force = np.zeros(2)

class Simulation:
    """
    Brute-force calculation of pairwise Coulomb interactions.
    Uses the kick-drift-kick algorithm for time integration.
    """
    def __init__(self, bodies):
        # store a list of Body instances
        self.bodies = bodies

    def compute_forces(self):
        """Compute the Coulomb force on each body due to every other body."""
        # reset forces before accumulation
        for b in self.bodies:
            b.force = np.zeros(2)

        # loop over unique pairs (i, j) with i < j (avoids double counting)
        N = len(self.bodies)
        for i in range(N):
            for j in range(i + 1, N):
                b1 = self.bodies[i]
                b2 = self.bodies[j]

                # b2 -> b1
                diff = b1.position - b2.position
                r2 = np.dot(diff, diff)
                r_soft = np.sqrt(r2 + soft**2) # softening

                # 2D Coulomb force: F = k * q1 * q2 * r_vec / r^2
                f_vec = k * b1.charge * b2.charge * diff / (r_soft**2)

                # Newton's third law
                b1.force += f_vec
                b2.force -= f_vec

    def step(self):
        """
        Kick-drift-kick integration:
        1. Half-step velocity update: v += (F / m) * (dt/2)
        2. Full-step position update: x += v * dt
        3. Recompute forces at the new positions
        4. Half-step velocity update: v += (F / m) * (dt/2)
        """
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt  # (1)
            b.position += b.velocity * dt                # (2)

        self.compute_forces() # (3)

        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt # (4)

class SimulationVectorised:
    """
    Vectorised implementation of the above using NumPy broadcasting.
    Reduces Python loops by operating on whole arrays.
    """
    def __init__(self, pos, vel, charge, mass=None):
        # pos: array of shape (N, 2), vel: array of shape (N, 2), charge: array of length N
        self.pos = np.array(pos, dtype=float) # (N,2)
        self.vel = np.array(vel, dtype=float) # (N,2)
        self.charge = np.array(charge, dtype=float) # (N,)
        self.N = self.pos.shape[0]
        self.force = np.zeros_like(self.pos) # (N,2)
        self.mass = np.ones(self.N) if mass is None else np.array(mass, dtype=float)

    def compute_forces(self):
        """
        Compute forces without explicit loops"""
        # pairwise position differences: (N, N, 2)
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        # squared distances plus softening: (N, N)
        r2 = np.sum(diff * diff, axis=2) + soft**2
        # inverse squared distances
        inv_r2 = 1.0 / r2
        # zero diagonal to remove self-force
        np.fill_diagonal(inv_r2, 0.0)

        # outer product of charges: (N, N)
        qq = np.outer(self.charge, self.charge)
        # force contributions tensor: (N, N, 2)
        Fmat = k * qq[:, :, None] * diff * inv_r2[:, :, None]
        # sum over j to get net force on each i: (N, 2)
        self.force = np.sum(Fmat, axis=1)

    def step(self):
        """ kick-drift-kick as before """
        self.vel += 0.5 * (self.force / self.mass[:, None]) * dt # (1)
        self.pos += self.vel * dt # (2)
        self.compute_forces() # (3)
        self.vel += 0.5 * (self.force / self.mass[:, None]) * dt # (4)