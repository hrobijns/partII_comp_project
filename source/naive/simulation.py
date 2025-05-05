import numpy as np

k = 1.0      # Coulomb constant
soft = 0.1  # softening length
dt = 0.01    # time step 

class Body:
    def __init__(self, position, velocity, charge, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge = charge
        self.mass = mass
        self.force = np.zeros(2)

class Simulation:
    def __init__(self, bodies):
        self.bodies = bodies

    def compute_forces(self):
        for b in self.bodies:
            b.force = np.zeros(2)
        N = len(self.bodies)
        for i in range(N):
            for j in range(i + 1, N):
                b1 = self.bodies[i]
                b2 = self.bodies[j]
                diff = b1.position - b2.position
                r2 = np.dot(diff, diff)
                r_soft = np.sqrt(r2 + soft**2)
                f_vec = k * b1.charge * b2.charge * diff / (r_soft**2)
                b1.force += f_vec
                b2.force -= f_vec

    def step(self):
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt
            b.position += b.velocity * dt
        self.compute_forces()
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt

class SimulationVectorised:
    def __init__(self, pos, vel, charge, mass=None):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.charge = np.array(charge, dtype=float)
        self.N = self.pos.shape[0]
        self.force = np.zeros_like(self.pos)
        if mass is None:
            self.mass = np.ones(self.N)
        else:
            self.mass = np.array(mass, dtype=float)

    def compute_forces(self):
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        r2 = np.sum(diff * diff, axis=2) + soft**2
        inv_r2 = 1.0 / r2
        np.fill_diagonal(inv_r2, 0.0)
        qq = np.outer(self.charge, self.charge)
        Fmat = k * qq[:, :, None] * diff * inv_r2[:, :, None]
        self.force = np.sum(Fmat, axis=1)

    def step(self):
        self.vel += 0.5 * (self.force / self.mass[:, None]) * dt
        self.pos += self.vel * dt
        self.compute_forces()
        self.vel += 0.5 * (self.force / self.mass[:, None]) * dt