import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from simulation import SimulationVectorised

class Animation:
    def __init__(self, sim, steps=200, interval=50, xlim=(-10,10), ylim=(-10,10)):
        self.sim = sim
        self.steps = steps
        self.interval = interval
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_facecolor('white')
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.scat = self.ax.scatter(sim.pos[:,0], sim.pos[:,1], s=16, c='blue')
        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps, interval=self.interval, blit=True)

    def update(self, frame):
        self.sim.step()
        self.scat.set_offsets(self.sim.pos)
        return (self.scat,)

    def show(self):
        plt.show()

def main():
    np.random.seed(42)
    N = 20
    pos = np.random.uniform(-1,1,(N,2))
    vel = np.random.uniform(-0.05,0.05,(N,2))
    charge = np.ones(N)
    sim = SimulationVectorised(pos, vel, charge)
    anim = Animation(sim, steps=500, interval=30)
    anim.show()

if __name__ == '__main__':
    main()