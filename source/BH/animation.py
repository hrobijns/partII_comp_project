import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from simulation import Simulation, Body
from quadtree import k, soft, dt, theta


class Animation:
    def __init__(self, bodies, simulation, steps=500, interval=20):
        self.bodies     = bodies
        self.sim        = simulation
        self.steps      = steps
        self.interval   = interval
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_facecolor('black')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-space_size, space_size)
        self.ax.set_ylim(-space_size, space_size)
        self.scat = [
            self.ax.plot([], [], 'wo', markersize=3+3*np.log10(abs(b.charge)+1e-3))[0]
            for b in bodies
        ]
        self.ani = FuncAnimation(
            self.fig, self._update, frames=self.steps,
            interval=self.interval, blit=True
        )

    def _update(self, _):
        self.sim.step()
        for sc, b in zip(self.scat, self.bodies):
            sc.set_data(b.position[0], b.position[1])
        return self.scat

    def show(self):
        plt.show()

# --- Example usage ---
if __name__ == "__main__":
    np.random.seed(42)
    n = 50
    space_size = 5.0

    bodies = []
    for _ in range(n):
        pos = np.random.uniform(-space_size, space_size, 2)
        vel = np.random.uniform(-0.1, 0.1, 2)
        q   = np.random.choice([-1, 1]) * np.random.rand()
        bodies.append(Body(pos, vel, charge=q, mass=1.0))

    sim  = Simulation(bodies, space_size=space_size, theta=theta)
    anim = Animation(bodies, sim, steps=1000, interval=30)
    anim.show()