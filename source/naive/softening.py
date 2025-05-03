import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import imageio

class Simulation:
    """
    2D N‐body simulation with an attractive logarithmic potential:
        U = k * sum_{i<j} q_i q_j * log(r_ij_soft)
        F_i = -∇_i U = -k * sum_{j!=i} q_i q_j * (r_i - r_j) / (r_ij_soft^2)
    """
    def __init__(self, pos, vel, charge, mass=1.0, k=1.0, soft=0.1):
        self.pos    = pos.copy()      # shape (N,2)
        self.vel    = vel.copy()      # shape (N,2)
        self.charge = charge          # shape (N,)
        self.mass   = mass
        self.k      = k
        self.soft   = soft
        self.N      = pos.shape[0]
        # initial forces & energy
        self.force = np.zeros_like(self.pos)
        self.compute_forces()
        self.E0    = self.total_energy()

    def compute_forces(self):
        self.force.fill(0.0)
        for i in range(self.N):
            for j in range(i+1, self.N):
                diff   = self.pos[i] - self.pos[j]
                r2     = np.dot(diff, diff)
                r_soft = np.sqrt(r2 + self.soft**2)
                # Attractive force toward each other
                f = - self.k * self.charge[i] * self.charge[j] * diff / (r_soft**2)
                self.force[i] += f
                self.force[j] -= f

    def step(self, dt):
        # velocity Verlet integrator
        self.vel += 0.5 * (self.force / self.mass) * dt
        self.pos += self.vel * dt
        self.compute_forces()
        self.vel += 0.5 * (self.force / self.mass) * dt

    def kinetic_energy(self):
        return 0.5 * self.mass * np.sum(self.vel**2)

    def potential_energy(self):
        U = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                diff   = self.pos[i] - self.pos[j]
                r_soft = np.sqrt(np.dot(diff, diff) + self.soft**2)
                U += self.k * self.charge[i] * self.charge[j] * np.log(r_soft)
        return U

    def total_energy(self):
        return self.kinetic_energy() + self.potential_energy()


class MultiAnimation:
    """
    Left: scatter of one simulation (first soften value).
    Right: evolving %ΔE curves for all softenings, plotted vs. step number.
    """
    def __init__(self, sims, softenings, dt, steps=200, interval=50,
                 xlim=(-2,2), ylim=(-2,2)):
        self.sims      = sims
        self.soft      = softenings
        self.dt        = dt
        self.steps     = steps

        # history array: percent ΔE for each sim at each step
        self.Ehist     = np.zeros((len(sims), steps))
        # x‐axis = step index
        self.steps_idx = np.arange(steps)

        # Set up figure with two subplots side by side
        self.fig, (self.ax_sim, self.ax_E) = plt.subplots(
            1, 2, figsize=(10,5), gridspec_kw={'width_ratios':[1,1.2]}
        )

        # --- Left: particle scatter ---
        self.ax_sim.set_xlim(*xlim)
        self.ax_sim.set_ylim(*ylim)
        self.ax_sim.set_xticks([]); self.ax_sim.set_yticks([])
        self.scat = self.ax_sim.scatter(
            sims[0].pos[:,0], sims[0].pos[:,1], s=20, c='blue'
        )

        # --- Right: energy drift plot ---
        # create one line per softening
        self.lines = []
        for s in self.soft:
            line, = self.ax_E.plot([], [], label=f"ε={s}")
            self.lines.append(line)

        # horizontal reference line at ΔE = 0
        self.ax_E.axhline(0.0, color='black', linestyle=':')

        # configure axes
        self.ax_E.set_xlim(0, self.steps - 1)
        self.ax_E.set_ylim(-5, 5)      # adjust to your expected drift range
        self.ax_E.set_xlabel("step")
        self.ax_E.set_ylabel("% ΔE")
        self.ax_E.legend(loc='upper right')
        self.ax_E.grid(True)

        # set up animation
        self.ani = FuncAnimation(
            self.fig, self.update, frames=self.steps,
            interval=interval, blit=False
        )

    def update(self, frame):
        # advance each sim one step and record its %ΔE
        for i, sim in enumerate(self.sims):
            sim.step(self.dt)
            Ei = sim.total_energy()
            self.Ehist[i, frame] = (Ei - sim.E0) / sim.E0 * 100

        # update particle positions for the first sim
        self.scat.set_offsets(self.sims[0].pos)

        # update each ΔE curve (x = step index)
        for i, line in enumerate(self.lines):
            line.set_data(self.steps_idx[:frame+1], self.Ehist[i, :frame+1])

        return [self.scat, *self.lines]

    def show(self):
        plt.tight_layout()
        plt.show()


def main():
    np.random.seed(3)
    N         = 3
    dt        = 0.01
    steps     = 301

    # initialize particles
    pos       = np.random.uniform(-1, 1, (N, 2))
    vel       = np.random.uniform(-0.5, 0.5, (N, 2))
    charge    = np.ones(N)

    # choose softenings to compare
    softenings = [0.2, 0.3, 0.4]
    sims = [
        Simulation(pos, vel, charge, mass=1.0, k=1.0, soft=s)
        for s in softenings
    ]

    anim = MultiAnimation(sims, softenings, dt, steps=steps, interval=3)
    #writer = FFMpegWriter(fps=60, codec='h264')
    #anim.ani.save("figures/simulation.mp4", writer=writer)

    writer = PillowWriter(fps=300)  
    anim.ani.save("figures/softening.gif", writer=writer)

    anim.show()


if __name__ == '__main__':
    main()
