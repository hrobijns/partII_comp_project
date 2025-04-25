import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
k    = 1.0      # Coulomb constant
soft = 1e-1     # softening length
dt   = 1/24     # time step (in days)

class Simulation:
    def __init__(self, pos, vel, charge, mass=None):
        """
        pos:    (N,2) array of initial positions
        vel:    (N,2) array of initial velocities
        charge: (N,)   array of charges
        mass:   (N,)   array of masses (defaults to 1.0)
        """
        self.pos    = pos.astype(float)
        self.vel    = vel.astype(float)
        self.charge = charge.astype(float)
        self.N      = pos.shape[0]
        self.force  = np.zeros_like(self.pos)
        if mass is None:
            self.mass = np.ones(self.N)
        else:
            self.mass = mass.astype(float)

    def compute_forces(self):
        """
        Compute pairwise repulsive 2D‐log (∝1/r) forces between all bodies,
        using a softened potential to avoid singularities.
        """
        # pairwise displacement vectors: shape (N, N, 2)
        diff = self.pos[:, None, :] - self.pos[None, :, :]

        # squared distances with softening: shape (N, N)
        r2 = np.sum(diff * diff, axis=2) + soft**2

        # inverse-square law (2D log potential): 1 / r^2
        inv_r2 = 1.0 / r2

        # eliminate self‐force
        np.fill_diagonal(inv_r2, 0.0)

        # pairwise charge products: shape (N, N)
        qq = np.outer(self.charge, self.charge)

        # build full force tensor: shape (N, N, 2)
        # F_ij = k * q_i q_j * diff_ij / (r^2)
        Fmat = k * qq[:, :, None] * diff * inv_r2[:, :, None]

        # net force on each particle: sum over j
        self.force = np.sum(Fmat, axis=1)  # shape (N, 2)

    def step(self):
        # half‐kick
        self.vel += 0.5 * (self.force / self.mass[:,None]) * dt
        # drift
        self.pos += self.vel * dt
        # recompute forces
        self.compute_forces()
        # half‐kick
        self.vel += 0.5 * (self.force / self.mass[:,None]) * dt

class Animation:
    def __init__(self, sim, steps=200, interval=50):
        self.sim   = sim
        self.steps = steps
        self.interval = interval

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # single scatter, faster than many plot() calls
        self.scat = self.ax.scatter(
            sim.pos[:,0], sim.pos[:,1],
            s=16, c='white'
        )

        self.ani = FuncAnimation(
            self.fig, self.update,
            frames=self.steps,
            interval=self.interval,
            blit=True
        )

    def update(self, frame):
        self.sim.step()
        # update all positions in one go
        self.scat.set_offsets(self.sim.pos)
        return (self.scat,)

    def show(self):
        plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    N = 20

    # initialize contiguous arrays
    pos    = np.random.uniform(-1,1,(N,2))
    vel    = np.random.uniform(-0.05,0.05,(N,2))
    charge = np.ones(N) * 1.0
    # mass defaults to 1

    sim  = Simulation(pos, vel, charge)
    anim = Animation(sim, steps=500, interval=30)
    anim.show()