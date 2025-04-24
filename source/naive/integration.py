import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy

# Gravitational constant in AU^3 M_sun^-1 day^-2
G = 2.959122082855911e-4  
dt = 1/24  # timestep (1 hour)


class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(2)


class Simulation:
    def __init__(self, bodies):
        self.bodies = bodies

    def compute_forces(self, e=1e-1):
        for b in self.bodies:
            b.force[:] = 0.0
        N = len(self.bodies)
        for i in range(N):
            for j in range(i+1, N):
                b1, b2 = self.bodies[i], self.bodies[j]
                diff = b2.position - b1.position
                r2 = np.dot(diff, diff) + e**2
                r = np.sqrt(r2)
                F = G * b1.mass * b2.mass / r2
                fvec = F * diff / r
                b1.force +=  fvec
                b2.force += -fvec

    def move_leapfrog(self):
        # kick
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt
        # drift
        for b in self.bodies:
            b.position += b.velocity * dt
        # recompute forces
        self.compute_forces()
        # kick
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt

    def move_euler(self):
        # forward Euler: x_{n+1} = x_n + v_n dt
        #                v_{n+1} = v_n + (F_n / m) dt
        # assumes self.bodies[].force has F_n
        for b in self.bodies:
            b.position += b.velocity * dt
        for b in self.bodies:
            b.velocity += (b.force / b.mass) * dt
        # update forces for next step
        self.compute_forces()

    def total_energy(self, e=1e-1):
        # kinetic
        K = sum(0.5 * b.mass * np.dot(b.velocity, b.velocity)
                for b in self.bodies)
        # potential
        U = 0.0
        N = len(self.bodies)
        for i in range(N):
            for j in range(i+1, N):
                b1, b2 = self.bodies[i], self.bodies[j]
                r2 = np.sum((b1.position - b2.position)**2) + e**2
                U -= G * b1.mass * b2.mass / np.sqrt(r2)
        return K + U


class Animation:
    def __init__(self, bodies, steps=500, interval=50):
        # make two independent copies
        bodies_lf = [copy.deepcopy(b) for b in bodies]
        bodies_eu = [copy.deepcopy(b) for b in bodies]

        self.sim_lf = Simulation(bodies_lf)
        self.sim_eu = Simulation(bodies_eu)
        self.sim_lf.compute_forces()
        self.sim_eu.compute_forces()

        self.steps = steps
        self.interval = interval

        # set up figure
        self.fig, (self.ax_anim, self.ax_metrics) = plt.subplots(1, 2, figsize=(12, 5))
        self.ax_anim.set_facecolor("black")
        self.ax_anim.set_xlim(-5, 5)
        self.ax_anim.set_ylim(-5, 5)
        self.ax_anim.set_xticks([])
        self.ax_anim.set_yticks([])

        # two sets of points: white = leapfrog, red = Euler
        self.scat_lf = [self.ax_anim.plot([], [], "wo", ms=b.mass*2)[0]
                        for b in bodies_lf]
        self.scat_eu = [self.ax_anim.plot([], [], "ro", ms=b.mass*2)[0]
                        for b in bodies_eu]

        # Energy tracking for both methods
        self.energy_steps = []
        self.energy_lf = []
        self.energy_eu = []
        self.line_lf, = self.ax_metrics.plot([], [], "r-", label="Leapfrog Δ%")
        self.line_eu, = self.ax_metrics.plot([], [], "g--", label="Euler Δ%")

        self.initial_E_lf = None
        self.initial_E_eu = None

        self.ax_metrics.set_title("Energy Conservation (% Change)")
        self.ax_metrics.set_xlabel("Step")
        self.ax_metrics.set_ylabel("% Change")
        self.ax_metrics.set_xlim(0, self.steps)
        self.ax_metrics.set_ylim(-1, 1)

        self.ax_metrics.legend()
        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps,
                                 interval=self.interval, repeat=False)

    def update(self, frame):
        # step both simulations
        self.sim_lf.move_leapfrog()
        self.sim_eu.move_euler()

        # update scatter positions
        for sc, b in zip(self.scat_lf, self.sim_lf.bodies):
            sc.set_data(b.position[0], b.position[1])
        for sc, b in zip(self.scat_eu, self.sim_eu.bodies):
            sc.set_data(b.position[0], b.position[1])

        # compute energies
        E_lf = self.sim_lf.total_energy()
        E_eu = self.sim_eu.total_energy()

        if frame == 0:
            self.initial_E_lf = E_lf
            self.initial_E_eu = E_eu
            pct_lf = pct_eu = 0.0
            # draw horizontal zero line
            self.ax_metrics.axhline(0, color="k", ls="dotted")
        else:
            pct_lf = 100 * (E_lf - self.initial_E_lf) / abs(self.initial_E_lf)
            pct_eu = 100 * (E_eu - self.initial_E_eu) / abs(self.initial_E_eu)

        # record
        self.energy_steps.append(frame)
        self.energy_lf.append(pct_lf)
        self.energy_eu.append(pct_eu)

        # update metric lines
        self.line_lf.set_data(self.energy_steps, self.energy_lf)
        self.line_eu.set_data(self.energy_steps, self.energy_eu)

        # rescale y
        all_vals = self.energy_lf + self.energy_eu
        mn, mx = min(all_vals), max(all_vals)
        pad = (mx - mn) * 0.1 if mx != mn else 1
        self.ax_metrics.set_ylim(mn - pad, mx + pad)

        return self.scat_lf + self.scat_eu + [self.line_lf, self.line_eu]

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    # random initial bodies
    bodies = [
        Body(
            position=np.random.uniform(-5, 5, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            mass=np.random.uniform(0.1, 1),
        )
        for _ in range(100)
    ]

    anim = Animation(bodies, steps=1000, interval=30)
    anim.show()
